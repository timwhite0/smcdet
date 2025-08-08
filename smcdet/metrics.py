import torch
from bliss.catalog import convert_nmgy_to_mag
from einops import rearrange
from scipy.optimize import linear_sum_assignment


def match_catalogs(
    true_counts,
    true_locs,
    true_fluxes,
    est_counts,
    est_locs,
    est_fluxes,
    num_est_catalogs_to_match,
    locs_tol,
    mags_tol,
    mag_bins,
):
    num_tiles = true_counts.shape[0]

    num_true_total_bucketed = torch.zeros(
        num_tiles, num_est_catalogs_to_match, len(mag_bins)
    )
    num_true_matches_bucketed = torch.zeros(
        num_tiles, num_est_catalogs_to_match, len(mag_bins)
    )
    num_est_total_bucketed = torch.zeros(
        num_tiles, num_est_catalogs_to_match, len(mag_bins)
    )
    num_est_matches_bucketed = torch.zeros(
        num_tiles, num_est_catalogs_to_match, len(mag_bins)
    )

    for t in range(num_tiles):
        true_locs_t = true_locs[t][: true_counts[t].int()]
        true_mags_t = convert_nmgy_to_mag(true_fluxes[t][: true_counts[t].int()])

        index = torch.randint(0, est_counts[t].shape[0], [num_est_catalogs_to_match])
        est_locs_t = est_locs[t][index]
        est_fluxes_t = est_fluxes[t][index]
        est_counts_t = est_counts[t][index].int()

        for n in range(num_est_catalogs_to_match):
            est_locs_t_n = est_locs_t[n][: est_counts_t[n]]
            est_mags_t_n = convert_nmgy_to_mag(est_fluxes_t[n][: est_counts_t[n]])

            locs_diff = rearrange(true_locs_t, "i j -> i 1 j") - rearrange(
                est_locs_t_n, "i j -> 1 i j"
            )
            locs_dist = locs_diff.norm(dim=-1)
            oob = locs_dist > locs_tol

            mags_dist = (
                rearrange(true_mags_t, "k -> k 1") - rearrange(est_mags_t_n, "k -> 1 k")
            ).abs()
            oob |= mags_dist > mags_tol

            cost = locs_dist + oob * 1e20
            row_indx, col_indx = linear_sum_assignment(cost)
            matches = ~oob[row_indx, col_indx].numpy()

            assert row_indx[matches].shape[0] == col_indx[matches].shape[0]
            true_matches = torch.from_numpy(row_indx[matches])
            est_matches = torch.from_numpy(col_indx[matches])

            true_mags_bucketed = torch.bucketize(true_mags_t, mag_bins).unsqueeze(
                -1
            ) == torch.arange(len(mag_bins))
            est_mags_bucketed = torch.bucketize(est_mags_t_n, mag_bins).unsqueeze(
                -1
            ) == torch.arange(len(mag_bins))

            num_true_total_bucketed[t, n] = true_mags_bucketed.sum(0)
            num_true_matches_bucketed[t, n] = true_mags_bucketed[true_matches].sum(0)
            num_est_total_bucketed[t, n] = est_mags_bucketed.sum(0)
            num_est_matches_bucketed[t, n] = est_mags_bucketed[est_matches].sum(0)

    return (
        num_true_total_bucketed,
        num_true_matches_bucketed,
        num_est_total_bucketed,
        num_est_matches_bucketed,
    )


def compute_precision_recall_f1(true_total, true_matches, est_total, est_matches):
    precision = (est_matches.sum(0) / est_total.sum(0)).nan_to_num(0)
    recall = (true_matches.sum(0) / true_total.sum(0)).nan_to_num(0)
    f1 = ((2 * precision * recall) / (precision + recall)).nan_to_num(0)

    return precision, recall, f1
