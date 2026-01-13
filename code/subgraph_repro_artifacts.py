import pandas as pd
u = "elboustany"
start="2017-01-01"; end="2017-12-31"
def load(path,label):
    df=pd.read_csv(path,usecols=["source","target","timestamp"])
    df["datetime"]=pd.to_datetime(df["timestamp"],unit="s",errors="coerce")
    df=df[df["source"].astype(str).str.strip()==u].dropna(subset=["datetime"])
    df=df[(df["datetime"]>=start)&(df["datetime"]<=end)].copy()
    df["month"]=df["datetime"].dt.to_period("M").astype(str)
    return df.groupby("month").size().rename(label)

h=load("hashtags_2017.csv","hashtag_events")
m=load("mentions_2017.csv","mention_events")
out=pd.concat([h,m],axis=1).fillna(0).astype(int)
out["union_events"]=out.sum(axis=1)
print(out)
print("\nmonths with union_events==0:", out.index[out["union_events"]==0].tolist())
