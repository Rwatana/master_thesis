import mlflow
from mlflow.tracking import MlflowClient
import os

def check_database(db_path):
    # DBファイルのフルパスを取得
    full_path = os.path.abspath(db_path)
    if not os.path.exists(full_path):
        print(f"❌ 指定されたDBファイルが見つかりません: {full_path}")
        return

    # URI形式に変換 (sqlite:///パス)
    db_uri = f"sqlite:///{full_path}"
    print(f"\n--- データベース診断: {db_uri} ---")
    
    # MLflowをこのDBに接続
    mlflow.set_tracking_uri(db_uri)
    client = MlflowClient()

    try:
        experiments = client.search_experiments()
        if not experiments:
            print("  (実験データは空です)")
            return

        for exp in experiments:
            print(f"\n📁 実験名: {exp.name} (ID: {exp.experiment_id})")
            
            # この実験内のランを取得
            runs = client.search_runs(exp.experiment_id)
            if not runs:
                print("    └ ランが見つかりません")
                continue
                
            for run in runs:
                run_name = run.data.tags.get('mlflow.runName', 'None')
                print(f"    └ 🚀 Run: {run_name}")
                print(f"       ID: {run.info.run_id} | Status: {run.info.status}")

        # ターゲットのRun IDを探す
        target_id = "e2f97f1507f74833a20417fdbde91bee"
        try:
            target_run = client.get_run(target_id)
            print(f"\n🎯 ターゲットID '{target_id}' をこのDB内で発見しました！")
        except:
            pass

    except Exception as e:
        print(f"⚠️ エラーが発生しました: {e}")

if __name__ == "__main__":
    # ファイル名が異なる場合はここを修正してください (例: 'my_experiments.db' など)
    target_db = "mlflow.db" 
    check_database(target_db)