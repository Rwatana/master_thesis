from transformers import pipeline
from PIL import Image

# 1. オブジェクト検出用のパイプラインを準備
#    (初回実行時にモデルのダウンロードが自動的に行われます)
try:
    object_detector = pipeline(
        "object-detection", 
        model="facebook/detr-resnet-50"
    )
    print("モデルのロードが完了しました。")
except Exception as e:
    print(f"モデルのロード中にエラーが発生しました: {e}")
    print("インターネット接続を確認するか、Hugging Face Hubのステータスを確認してください。")
    exit()

# 2. 判定したい画像を読み込む
image_path = "organized_images/beauty/_alussier/_alussier-1377753729083757567.jpg"  # ここに判定したいJPGファイルパスを指定
try:
    img = Image.open(image_path)
    print(f"画像を読み込みました: {image_path}")
except FileNotFoundError:
    print(f"エラー: 画像ファイルが見つかりません: {image_path}")
    exit()

# 3. オブジェクト検出を実行
try:
    results = object_detector(img)

    # 4. 結果の表示
    if not results:
        print("画像から物体は検出されませんでした。")
    else:
        print("\n--- 検出結果 ---")
        for i, result in enumerate(results):
            label = result['label']
            score = result['score']
            box = result['box']
            
            print(f"物体 {i+1}:")
            print(f"  ラベル (Label): {label}")
            print(f"  信頼度 (Score): {score:.4f}") # 信頼度を小数点以下4桁で表示
            print(f"  位置 (Box): x_min={box['xmin']}, y_min={box['ymin']}, x_max={box['xmax']}, y_max={box['ymax']}")

except Exception as e:
    print(f"オブジェクト検出の実行中にエラーが発生しました: {e}")
