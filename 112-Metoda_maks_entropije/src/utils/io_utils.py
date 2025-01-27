import numpy as np
from config import DATA_ROOT

def load_data(data_name):
    data = np.loadtxt(DATA_ROOT + data_name + ".dat")
    return data


def load_co2(data_name):
    data = np.loadtxt(DATA_ROOT + data_name + ".dat")
    data = data[~(data[:,1] == -99.99),:]
    print(np.shape(data))
    return data


def load_Wolf(data_name):    
    data = np.loadtxt(DATA_ROOT + data_name + ".dat")
    t = [year + (month - 1) / 12 for year, month, value in data]
    data = np.column_stack((t, data[:, 2]))
    print(np.shape(data))
    return data


def load_lunar_data(data_name):

    def date_to_decimal_year(date_str):
        year = 1995  # Osnovno leto
        month_days = {
            "JAN": 0, "FEB": 31, "MAR": 59, "APR": 90, "MAY": 120, "JUN": 151,
            "JUL": 181, "AUG": 212, "SEP": 243, "OCT": 273, "NOV": 304, "DEC": 334
        }
        month = date_str[:3]  # Prvi trije znaki so mesec
        day = int(date_str[3:])  # Zadnji del je dan
        decimal_year = year + (month_days[month] + day - 1) / 365.0  # Pretvorba v decimalno leto
        return decimal_year

    data = np.loadtxt(
        DATA_ROOT + data_name + ".dat",
        dtype={"names": ("datum", "RA", "Dec"), "formats": ("U5", "f8", "f8")},
        skiprows=2
    )

    t = np.array([date_to_decimal_year(d) for d in data["datum"]])
    RA = data["RA"]
    Dec = data["Dec"]
    data_Dec = np.column_stack((t, Dec))
    data_RA = np.column_stack((t, RA))
    return data_Dec, data_RA