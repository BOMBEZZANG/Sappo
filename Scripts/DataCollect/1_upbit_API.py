import pyupbit
import pandas as pd
import time
from datetime import datetime
import os

def get_upbit_data(coin_ticker, interval, start_date_limit, output_dir):
    """
    업비트 API 공식 문서를 기반으로 특정 코인의 데이터를 수집하여 CSV 파일로 저장
    """
    try:
        all_data = []
        to_time = datetime.utcnow()
        last_to_time = None # <--- 추가: 이전 요청 시간을 저장할 변수

        print(f"{coin_ticker} ({interval}): 데이터 수집 시작... (목표 시작일: {start_date_limit.strftime('%Y-%m-%d')})")

        while True:
            # <--- 추가: 무한 루프 방지 코드
            if to_time == last_to_time:
                print("  -> 데이터가 더 이상 변경되지 않아 수집을 중단합니다.")
                break
            
            last_to_time = to_time # 현재 요청 시간을 저장

            df = pyupbit.get_ohlcv(ticker=coin_ticker, interval=interval, to=to_time, count=200)

            if df is None or df.empty:
                print(f"  -> {coin_ticker} ({interval}): 조회할 데이터가 더 이상 없습니다.")
                break
            
            oldest_time_in_df = df.index[0].to_pydatetime()
            latest_time_in_df = df.index[-1].to_pydatetime()
            print(f"  -> {oldest_time_in_df.strftime('%Y-%m-%d')} ~ {latest_time_in_df.strftime('%Y-%m-%d')} 데이터 수신 완료. 계속 진행합니다...")
            
            all_data.append(df)
            
            oldest_time = oldest_time_in_df.replace(tzinfo=None)
            
            if oldest_time < start_date_limit:
                print(f"  -> {coin_ticker} ({interval}): 목표 시작일에 도달하여 수집을 종료합니다.")
                break
            
            to_time = oldest_time
            time.sleep(0.2)
        
        if all_data:
            combined_df = pd.concat(all_data)
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            combined_df = combined_df.sort_index()
            combined_df = combined_df[combined_df.index >= start_date_limit]
            
            os.makedirs(output_dir, exist_ok=True)
            coin_name = coin_ticker.split('-')[1]
            
            interval_name_map = {"day": "day", "week": "week", "minute60": "hour1", "minute240": "hour4"}
            interval_name = interval_name_map.get(interval, interval)
            output_path = os.path.join(output_dir, f"upbit_{coin_name}_{interval_name}.csv")
            
            combined_df.reset_index().rename(columns={'index': 'date'}).to_csv(output_path, index=False)
            print(f"✅ {coin_ticker} ({interval}) 데이터 저장 완료: {output_path}")
            return output_path
        else:
            print(f"ℹ️ {coin_ticker} ({interval})에 대한 데이터를 찾을 수 없습니다.")
            return None
            
    except Exception as e:
        print(f"❌ {coin_ticker} ({interval}) 데이터 수집 중 오류 발생: {e}")
        return None

def collect_upbit_data():
    """
    업비트 API를 사용하여 주요 코인들의 여러 시간봉 데이터 수집
    """
    # -------------------------------------------------------------------
    # ⚠️ 매우 중요한 보안 경고 ⚠️
    # 실제 운영 코드에는 아래와 같이 API 키를 직접 작성하지 마세요.
    # 이 코드를 GitHub 등에 올리면 자산이 탈취될 위험이 매우 큽니다.
    # os.environ.get('UPBIT_ACCESS_KEY') 와 같이 환경 변수를 사용하는 것을 권장합니다.

    
    # 'Upbit' 객체는 향후 주문, 잔고 조회 등 개인 API 호출 시 사용됩니다.
    # upbit = pyupbit.Upbit(access_key, secret_key)
    # 예시: my_balance = upbit.get_balance("KRW")
    # -------------------------------------------------------------------

    coins = [
        "KRW-ICX", "KRW-CTC", "KRW-BORA", "KRW-DKA", "KRW-PDA", "KRW-TON", 
        "KRW-MLK", "KRW-MVL", "KRW-MED", "KRW-BFC", "KRW-HUNT", "KRW-HPO", 
        "KRW-STMX", "KRW-META", "KRW-CRE", "KRW-AERGO", "KRW-SSX", "KRW-CBK", 
        "KRW-FCT2", "KRW-MOC", "KRW-AQT", "KRW-MBL", "KRW-UPP", "KRW-AHT", 
        "KRW-ONIT", "KRW-QTCON", "KRW-OBSR"
    ]
    
    intervals = ["day", "week", "minute240", "minute60"]
    start_date_limit = datetime(2017, 1, 1) 
    output_dir = r"C:\Projects\Alt_Bubble\data\upbit_data"
    
    for coin in coins:
        for interval in intervals:
            # get_upbit_data 함수는 공개 API를 사용하므로 API 키가 필요 없습니다.
            get_upbit_data(coin, interval, start_date_limit, output_dir)
            time.sleep(1)

def main():
    collect_upbit_data()
    print("\n모든 업비트 데이터 수집이 완료되었습니다!")

if __name__ == "__main__":
    main()