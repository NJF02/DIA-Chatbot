from pathlib import Path

dir = str(Path(__file__).parent)

train_path = dir + "\\Train\\"
data_path = dir + "\\Data\\"
categories = ["general", "menu_cuisine", "reservation", "delivery"]

Path(data_path).mkdir(parents = True, exist_ok = True)