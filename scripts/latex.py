from sys import stdin, stdout

import pandas as pd


def main():
    df = pd.read_csv(stdin, index_col=0)

    df.to_latex(stdout)


if __name__ == '__main__':
    main()
