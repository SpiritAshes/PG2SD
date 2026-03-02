import os

import numpy as np
import cv2
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

class Evaluator(object):

    def __init__(self):
        self.mutual_check = True
        self.err_thld = np.arange(1, 16)  # range [1,15]
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu:0')
        self.stats = {
            'i_eval_stats': np.zeros((len(self.err_thld), 8), np.float32),
            'v_eval_stats': np.zeros((len(self.err_thld), 8), np.float32),
            'all_eval_stats': np.zeros((len(self.err_thld), 8), np.float32),
        }

    def homo_trans(self, coord, H):
        kpt_num = coord.shape[0]
        homo_coord = np.concatenate((coord, np.ones((kpt_num, 1))), axis=-1)
        proj_coord = np.matmul(H, homo_coord.T).T
        proj_coord = proj_coord / proj_coord[:, 2][..., None]
        proj_coord = proj_coord[:, 0:2]
        return proj_coord

    def mnn_matcher(self, descriptors_a, descriptors_b):
        descriptors_a = torch.from_numpy(descriptors_a).to(self.device)
        descriptors_b = torch.from_numpy(descriptors_b).to(self.device)
        sim = descriptors_a @ descriptors_b.t()
        nn12 = torch.max(sim, dim=1)[1]
        nn21 = torch.max(sim, dim=0)[1]
        ids1 = torch.arange(0, sim.shape[0], device=self.device)
        mask = (ids1 == nn21[nn12])
        matches = torch.stack([ids1[mask], nn12[mask]])
        return matches.t().detach().cpu().numpy()

    def feature_matcher(self, ref_feat, test_feat):
        matches = self.mnn_matcher(ref_feat, test_feat)
        matches = [cv2.DMatch(matches[i][0], matches[i][1], 0) for i in range(matches.shape[0])]
        return matches

    def get_covisible_mask(self, ref_coord, test_coord, ref_img_shape, test_img_shape, gt_homo, scaling=1.):
        ref_coord = ref_coord / scaling
        test_coord = test_coord / scaling

        proj_ref_coord = self.homo_trans(ref_coord, gt_homo)
        proj_test_coord = self.homo_trans(test_coord, np.linalg.inv(gt_homo))

        ref_mask = np.logical_and(
            np.logical_and(proj_ref_coord[:, 0] < test_img_shape[1] - 1,
                           proj_ref_coord[:, 1] < test_img_shape[0] - 1),
            np.logical_and(proj_ref_coord[:, 0] > 0, proj_ref_coord[:, 1] > 0)
        )

        test_mask = np.logical_and(
            np.logical_and(proj_test_coord[:, 0] < ref_img_shape[1] - 1,
                           proj_test_coord[:, 1] < ref_img_shape[0] - 1),
            np.logical_and(proj_test_coord[:, 0] > 0, proj_test_coord[:, 1] > 0)
        )

        return ref_mask, test_mask

    def get_inlier_matches(self, ref_coord, test_coord, putative_matches, gt_homo, scaling=1.):
        p_ref_coord = np.float32([ref_coord[m.queryIdx] for m in putative_matches]) / scaling
        p_test_coord = np.float32([test_coord[m.trainIdx] for m in putative_matches]) / scaling

        proj_p_ref_coord = self.homo_trans(p_ref_coord, gt_homo)
        dist = np.sqrt(np.sum(np.square(proj_p_ref_coord - p_test_coord[:, 0:2]), axis=-1))
        inlier_matches_list = []
        for err_thld in self.err_thld:
            inlier_mask = dist <= err_thld
            inlier_matches = [putative_matches[z] for z in np.nonzero(inlier_mask)[0]]
            inlier_matches_list.append(inlier_matches)
        return inlier_matches_list

    def get_gt_matches(self, ref_coord, test_coord, gt_homo, scaling=1.):
        ref_coord = ref_coord / scaling
        test_coord = test_coord / scaling
        proj_ref_coord = self.homo_trans(ref_coord, gt_homo)

        pt0 = np.expand_dims(proj_ref_coord, axis=1)
        pt1 = np.expand_dims(test_coord, axis=0)
        norm = np.linalg.norm(pt0 - pt1, ord=None, axis=2)
        min_dist0 = np.min(norm, axis=1)
        min_dist1 = np.min(norm, axis=0)
        gt_num_list = []
        for err_thld in self.err_thld:
            gt_num0 = np.sum(min_dist0 <= err_thld)
            gt_num1 = np.sum(min_dist1 <= err_thld)
            gt_num = (gt_num0 + gt_num1) / 2
            gt_num_list.append(gt_num)
        return gt_num_list

    def compute_homography_accuracy(self, ref_coord, test_coord, ref_img_shape, putative_matches, gt_homo, scaling=1.):
        ref_coord = np.float32([ref_coord[m.queryIdx] for m in putative_matches])
        test_coord = np.float32([test_coord[m.trainIdx] for m in putative_matches])

        pred_homo, _ = cv2.findHomography(ref_coord, test_coord, cv2.RANSAC)
        if pred_homo is None:
            correctness_list = [0 for i in range(len(self.err_thld))]
        else:
            corners = np.array([[0, 0],
                                [ref_img_shape[1] - 1, 0],
                                [0, ref_img_shape[0] - 1],
                                [ref_img_shape[1] - 1, ref_img_shape[0] - 1]])
            real_warped_corners = self.homo_trans(corners, gt_homo)
            warped_corners = self.homo_trans(corners, pred_homo)
            mean_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))
            correctness_list = []
            for err_thld in self.err_thld:
                correctness = float(mean_dist <= err_thld)
                correctness_list.append(correctness)
        return correctness_list

    def print_stats(self, key):
        for i, err_thld in enumerate(self.err_thld):
            avg_stats = self.stats[key][i] / max(self.stats[key][i][0], 1)
            avg_stats = avg_stats[1:]
            print('----------%s----------' % key)
            print('threshold: %d' % err_thld)
            print('avg_n_feat', int(avg_stats[0]))
            print('avg_rep', avg_stats[1])
            print('avg_precision', avg_stats[2])
            print('avg_matching_score', avg_stats[3])
            print('avg_recall', avg_stats[4])
            print('avg_MMA', avg_stats[5])
            print('avg_homography_accuracy', avg_stats[6])

    def save_results(self, file):
        for i, err_thld in enumerate(self.err_thld):
            for key in ['i_eval_stats', 'v_eval_stats', 'all_eval_stats']:
                avg_stats = self.stats[key][i] / max(self.stats[key][i][0], 1)
                avg_stats = avg_stats[1:]
                file.write('----------%s----------\n' % key)
                file.write('threshold: %d\n' % err_thld)
                file.write('avg_n_feat: %d\n' % int(avg_stats[0]))
                file.write('avg_rep: %.4f\n' % avg_stats[1])
                file.write('avg_precision: %.4f\n' % avg_stats[2])
                file.write('avg_matching_score: %.4f\n' % avg_stats[3])
                file.write('avg_recall: %.4f\n' % avg_stats[4])
                file.write('avg_MMA: %.4f\n' % avg_stats[5])
                file.write('avg_homography_accuracy: %.4f\n' % avg_stats[6])


def evaluate(read_feats, dataset_path, evaluator):
    seq_names = sorted(os.listdir(dataset_path))

    for seq_idx, seq_name in tqdm(enumerate(seq_names), total=len(seq_names)):
        ref_kpts, ref_descs = read_feats(seq_name, 1)
        ref_img = cv2.imread(os.path.join(dataset_path, seq_name, '1.ppm'))
        ref_img_shape = ref_img.shape
        eval_stats = np.zeros((len(evaluator.err_thld), 8), np.float32)

        # print(seq_idx, seq_name)

        for im_idx in range(2, 7):
            test_kpts, test_descs = read_feats(seq_name, im_idx)
            gt_homo = np.loadtxt(os.path.join(dataset_path, seq_name, "H_1_" + str(im_idx)))
            test_img = cv2.imread(os.path.join(dataset_path, seq_name, str(im_idx) + '.ppm'))
            test_img_shape = test_img.shape
            # get MMA
            num_feat = min(ref_kpts.shape[0], test_kpts.shape[0])
            if num_feat > 0:
                mma_putative_matches = evaluator.feature_matcher(ref_descs, test_descs)
            else:
                mma_putative_matches = []
            mma_inlier_matches_list = evaluator.get_inlier_matches(ref_kpts, test_kpts, mma_putative_matches, gt_homo)
            num_mma_putative = len(mma_putative_matches)
            num_mma_inlier_list = [len(mma_inlier_matches) for mma_inlier_matches in mma_inlier_matches_list]

            # get covisible keypoints
            ref_mask, test_mask = evaluator.get_covisible_mask(ref_kpts, test_kpts,
                                                               ref_img_shape, test_img_shape,
                                                               gt_homo)
            cov_ref_coord, cov_test_coord = ref_kpts[ref_mask], test_kpts[test_mask]
            cov_ref_feat, cov_test_feat = ref_descs[ref_mask], test_descs[test_mask]
            num_cov_feat = (cov_ref_coord.shape[0] + cov_test_coord.shape[0]) / 2
            # num_cov_feat = min(cov_ref_coord.shape[0], cov_test_coord.shape[0])
            # get gt matches
            gt_num_list = evaluator.get_gt_matches(cov_ref_coord, cov_test_coord, gt_homo)
            # establish putative matches
            if num_cov_feat > 0:
                putative_matches = evaluator.feature_matcher(cov_ref_feat, cov_test_feat)
            else:
                putative_matches = []
            num_putative = max(len(putative_matches), 1)

            # get homography accuracy
            correctness_list = evaluator.compute_homography_accuracy(ref_kpts, test_kpts, ref_img_shape,
                                                                mma_putative_matches, gt_homo)
            # get inlier matches
            inlier_matches_list = evaluator.get_inlier_matches(cov_ref_coord, cov_test_coord, putative_matches, gt_homo)
            num_inlier_list = [len(inlier_matches) for inlier_matches in inlier_matches_list]

            eval_stats += np.stack([np.array((1,  # counter
                       num_feat,  # feature number
                       gt_num_list[i] / max(num_cov_feat, 1),  # repeatability
                       num_inlier_list[i] / max(num_putative, 1),  # precision
                       num_inlier_list[i] / max(num_cov_feat, 1),  # matching score
                       num_inlier_list[i] / max(gt_num_list[i], 1),  # recall
                       num_mma_inlier_list[i] / max(num_mma_putative, 1),
                       correctness_list[i])) / 5  # MHA
             for i in range(len(evaluator.err_thld))
             ], axis=0)  # [len(evaluator.err_thld), 8]

        # print(int(eval_stats[1]), eval_stats[2:])
        evaluator.stats['all_eval_stats'] += eval_stats
        if os.path.basename(seq_name)[0] == 'i':
            evaluator.stats['i_eval_stats'] += eval_stats
        if os.path.basename(seq_name)[0] == 'v':
            evaluator.stats['v_eval_stats'] += eval_stats

    evaluator.print_stats('i_eval_stats')
    evaluator.print_stats('v_eval_stats')
    evaluator.print_stats('all_eval_stats')

    err_thld = evaluator.err_thld
    i_eval_stats = evaluator.stats['i_eval_stats'].T  # [8, 15]
    i_eval_count = i_eval_stats[0, 0]
    i_err = {
        'Rep.': {thr: i_eval_stats[2][i] for i, thr in enumerate(err_thld)},
        'Precision': {thr: i_eval_stats[3][i] for i, thr in enumerate(err_thld)},
        'M.S.': {thr: i_eval_stats[4][i] for i, thr in enumerate(err_thld)},
        'MMA': {thr: i_eval_stats[6][i] for i, thr in enumerate(err_thld)},
        'HA': {thr: i_eval_stats[7][i] for i, thr in enumerate(err_thld)},
    }

    v_eval_stats = evaluator.stats['v_eval_stats'].T
    v_eval_count = v_eval_stats[0, 0]
    v_err = {
        'Rep.': {thr: v_eval_stats[2][i] for i, thr in enumerate(err_thld)},
        'Precision': {thr: v_eval_stats[3][i] for i, thr in enumerate(err_thld)},
        'M.S.': {thr: v_eval_stats[4][i] for i, thr in enumerate(err_thld)},
        'MMA': {thr: v_eval_stats[6][i] for i, thr in enumerate(err_thld)},
        'HA': {thr: v_eval_stats[7][i] for i, thr in enumerate(err_thld)},
    }

    return {
        'i_err': i_err,
        'i_count': i_eval_count,
        'v_err': v_err,
        'v_count': v_eval_count,
    }


def draw_point_match(read_feats, dataset_path, evaluator, path):
    seq_names = sorted(os.listdir(dataset_path))

    for seq_idx, seq_name in tqdm(enumerate(seq_names), total=len(seq_names)):
        ref_kpts, ref_descs = read_feats(seq_name, 1)
        ref_img = cv2.imread(os.path.join(dataset_path, seq_name, '1.ppm'))

        for im_idx in range(2, 7):
            test_kpts, test_descs = read_feats(seq_name, im_idx)
            gt_homo = np.loadtxt(os.path.join(dataset_path, seq_name, "H_1_" + str(im_idx)))
            test_img = cv2.imread(os.path.join(dataset_path, seq_name, str(im_idx) + '.ppm'))

            # get MMA
            num_feat = min(ref_kpts.shape[0], test_kpts.shape[0])
            if num_feat > 0:
                mma_putative_matches = evaluator.feature_matcher(ref_descs, test_descs)
            else:
                mma_putative_matches = []

            p_ref_coord = np.float32([ref_kpts[m.queryIdx] for m in mma_putative_matches]) / 1.0
            p_test_coord = np.float32([test_kpts[m.trainIdx] for m in mma_putative_matches]) / 1.0

            proj_p_ref_coord = evaluator.homo_trans(p_ref_coord, gt_homo)
            dist = np.sqrt(np.sum(np.square(proj_p_ref_coord - p_test_coord[:, 0:2]), axis=-1))

            inlier_mask = dist <= 3
            inlier_matches = [mma_putative_matches[z] for z in np.nonzero(inlier_mask)[0]]

            height = max(ref_img.shape[0], test_img.shape[0])
            weight = ref_img.shape[1] + test_img.shape[1] + 20
            match_image = np.ones((height, weight, 3), dtype=np.uint8) * 255
            match_image[:ref_img.shape[0], :ref_img.shape[1]] = ref_img
            match_image[:test_img.shape[0], ref_img.shape[1] + 20:] = test_img
            for point in ref_kpts:
                cv2.circle(match_image, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)
            for point in test_kpts:
                cv2.circle(match_image, (int(point[0]) + ref_img.shape[1], int(point[1])), 2, (0, 0, 255), -1)
            for match in inlier_matches:
                img1_idx = match.queryIdx
                img2_idx = match.trainIdx
                x1, y1 = ref_kpts[img1_idx]
                x2, y2 = test_kpts[img2_idx]
                cv2.line(match_image, (int(x1), int(y1)), (int(x2) + ref_img.shape[1], int(y2)), (0, 255, 0), 1)

            # cv2.imshow('Matches', match_image)
            # cv2.waitKey(0)

            save_path = os.path.join(path, f'matches_{seq_idx}{im_idx}.png')
            cv2.imwrite(save_path, match_image)
            
def obtain_different_thr_error(read_feats, dataset_path, evaluator):
    seq_names = sorted(os.listdir(dataset_path))
    lim = [1, 10]
    rng = np.arange(lim[0], lim[1] + 1)
    i_err = {thr: 0 for thr in rng}
    v_err = {thr: 0 for thr in rng}

    for seq_idx, seq_name in tqdm(enumerate(seq_names), total=len(seq_names)):
        ref_kpts, ref_descs = read_feats(seq_name, 1)
        ref_img = cv2.imread(os.path.join(dataset_path, seq_name, '1.ppm'))

        for im_idx in range(2, 7):
            test_kpts, test_descs = read_feats(seq_name, im_idx)
            gt_homo = np.loadtxt(os.path.join(dataset_path, seq_name, "H_1_" + str(im_idx)))
            test_img = cv2.imread(os.path.join(dataset_path, seq_name, str(im_idx) + '.ppm'))

            # get MMA
            num_feat = min(ref_kpts.shape[0], test_kpts.shape[0])
            if num_feat > 0:
                mma_putative_matches = evaluator.feature_matcher(ref_descs, test_descs)
            else:
                mma_putative_matches = []

            num_mma_putative = len(mma_putative_matches)

            p_ref_coord = np.float32([ref_kpts[m.queryIdx] for m in mma_putative_matches]) / 1.
            p_test_coord = np.float32([test_kpts[m.trainIdx] for m in mma_putative_matches]) / 1.

            proj_p_ref_coord = evaluator.homo_trans(p_ref_coord, gt_homo)
            dist = np.sqrt(np.sum(np.square(proj_p_ref_coord - p_test_coord[:, 0:2]), axis=-1))

            for err_thld in rng:
                inlier_mask = dist <= err_thld
                inlier_matches = [mma_putative_matches[z] for z in np.nonzero(inlier_mask)[0]]
                num_inlier_matches = len(inlier_matches)
                if seq_name[0] == 'i':
                    i_err[err_thld] += num_inlier_matches / max(num_mma_putative, 1)
                else:
                    v_err[err_thld] += num_inlier_matches / max(num_mma_putative, 1)
    return i_err, v_err

def draw_mma(dataset_path, dataset_path_d2, evaluator, cache_dir, top_k=None):
    methods = ['sp', 'd2-net', 'r2d2', 'caps_sp', 'r2d2', 'aslfeat', 'DISK_2000', 'MTLDesc', 'CNDesc_SP', 'alike-n-ms', 'AWDesc-T16', 'AWDesc-T32', 'PG2SD']
    names = ['SuperPoint', 'D2-Net', 'R2D2', 'CAPS+SuperPoint', 'R2D2', 'ASLFeat', 'DISK', 'MTLDesc', 'CNDesc+SuperPoint', 'ALIKE', 'AWDesc-T16', 'AWDesc-T32', 'PG2SD']
    colors = ['purple', 'blue', 'blue', 'brown', 'brown', 'purple', 'green', 'green', 'orange', 'orange', 'black', 'black', 'red']
    linestyles = ['-', '-', '--', '--', '-', '--', '-', '--', '--', '-', '--', '-', '-']


    n_i = 52
    n_v = 56

    errors = {}
    for method in methods:
        output_file = os.path.join(cache_dir, method + '.npy')

        read_function = generate_read_function(method, dataset_path, top_k)
        # read_function = caps_disk_generate_read_function('sp')
        if os.path.exists(output_file):
            errors[method] = np.load(output_file, allow_pickle=True)
        else:
            errors[method] = obtain_different_thr_error(read_function, dataset_path, evaluator)
            np.save(output_file, errors[method])

    for name, method in zip(names, methods):
        plt_lim = [1, 10]
        plt_rng = np.arange(plt_lim[0], plt_lim[1] + 1)
        plt.rc('axes', titlesize=25)
        plt.rc('axes', labelsize=25)

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        for method, name, color, ls in zip(methods, names, colors, linestyles):
            i_err, v_err = errors[method]
            plt.plot(plt_rng, [(i_err[thr] + v_err[thr]) / ((n_i + n_v) * 5) for thr in plt_rng], color=color, ls=ls, linewidth=3, label=name)
        plt.title('Overall')
        plt.xlim(plt_lim)
        plt.xticks(plt_rng)
        plt.ylabel('MMA')
        plt.ylim([0, 1])
        plt.grid()
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.legend()

        plt.subplot(1, 3, 2)
        for method, name, color, ls in zip(methods, names, colors, linestyles):
            i_err, v_err = errors[method]
            plt.plot(plt_rng, [i_err[thr] / (n_i * 5) for thr in plt_rng], color=color, ls=ls, linewidth=3, label=name)
        plt.title('Illumination')
        plt.xlabel('threshold [px]')
        plt.xlim(plt_lim)
        plt.xticks(plt_rng)
        plt.ylim([0, 1])
        plt.gca().axes.set_yticklabels([])
        plt.grid()
        plt.tick_params(axis='both', which='major', labelsize=20)

        plt.subplot(1, 3, 3)
        for method, name, color, ls in zip(methods, names, colors, linestyles):
            i_err, v_err = errors[method]
            plt.plot(plt_rng, [v_err[thr] / (n_v * 5) for thr in plt_rng], color=color, ls=ls, linewidth=3, label=name)
        plt.title('Viewpoint')
        plt.xlim(plt_lim)
        plt.xticks(plt_rng)
        plt.ylim([0, 1])
        plt.gca().axes.set_yticklabels([])
        plt.grid()
        plt.tick_params(axis='both', which='major', labelsize=20)

        if top_k is None:
            # plt.savefig('hseq.png', bbox_inches='tight', dpi=600)
            plt.savefig('hseq.pdf', bbox_inches='tight', dpi=600)
        else:
            plt.savefig('hseq-top.png', bbox_inches='tight', dpi=300)


def generate_read_function(method, path, top_k=None, extension='ppm'):
    def read_function(seq_name, im_idx):
        aux = np.load(os.path.join(path, seq_name, '%d.%s.%s' % (im_idx, extension, method)))
        if top_k is None:
            return aux['keypoints'][:, :2], aux['descriptors']
        else:
            assert ('scores' in aux)
            ids = np.argsort(aux['scores'])[-top_k:]
            return aux['keypoints'][ids, :], aux['descriptors'][ids, :]
    return read_function


if __name__ == '__main__':
    top_k = None
    data_path = 'hpatches_sequences/hpatches-sequences-release'
    cache_dir = 'cache_test' # './cache'

    read_function = generate_read_function('PG2SD', data_path)

    evaluate(read_function, data_path, Evaluator())

    # draw_point_match(read_function, data_path, Evaluator(), save_path)

    # draw_mma(data_path, data_path, Evaluator(), cache_dir, top_k)