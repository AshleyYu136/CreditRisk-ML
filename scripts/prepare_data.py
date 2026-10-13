from creditrisk.config import DATA
from creditrisk.data_loader import download, load_data, validate

if __name__ == "__main__":
    frame = load_data(download())
    (DATA / "processed").mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATA / "processed/clients.csv", index=False)
    print(validate(frame))
