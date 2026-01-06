# Template for Bachelor / Master's Thesis

村田先生が2020年2月に編集してくださったテンプレート

元のファイル: https://waseda.app.box.com/folder/103009238555

## To use
1. Clone this repository
2. Edit files
3. Use `latexmkrc` to compile

## 楽しい卒論・修論ライフのために
便利なtipsなどは[wiki](https://github.com/murata-lab/template-thesis/wiki)に書いてください

## Versions (environment snapshot)
- Python: 3.13.7
- torch: not_installed
- torch_geometric: not_installed
- pandas: not_installed
- numpy: not_installed
- scikit-learn: not_installed
- mlflow: not_installed
- matplotlib: not_installed


## 活動しているインフルエンサーの選定
```
python find_consistently_active_users.py \
  --end_date 2017-12-31 --num_months 12 \
  --min_posts 5 --min_active_days 3 \
  --topk 200

```

