def score_auc(clicks, shows, scores):
  rank_sorted = sorted(range(len(scores)),key=lambda i: scores[i], reverse=True)
  auc_temp = 0.0
  click_sum = 0.0
  old_click_sum = 0.0
  no_click = 0.0
  no_click_sum = 0.0
  last_ctr = scores[rank_sorted[0]] + 1.0
  for i in range(len(scores)):
      if last_ctr != scores[rank_sorted[i]]:
          auc_temp += (click_sum+old_click_sum) * no_click / 2.0
          old_click_sum = click_sum
          no_click = 0.0
          last_ctr = scores[rank_sorted[i]]
      no_click += shows[rank_sorted[i]] - clicks[rank_sorted[i]]
      no_click_sum += shows[rank_sorted[i]] - clicks[rank_sorted[i]]
      click_sum += clicks[rank_sorted[i]]
  auc_temp += (click_sum+old_click_sum) * no_click / 2.0
  auc = auc_temp / (click_sum * no_click_sum)