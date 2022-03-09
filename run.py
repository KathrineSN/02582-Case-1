from data import load
from fit import fit
from pelutils import log, Levels


if __name__ == "__main__":
    #log.configure("case1.log", print_level=Levels.DEBUG)
    with log.log_errors:
        df = load("Realized Schedule 20210101-20220208.xlsx")
        fit(df, 5)