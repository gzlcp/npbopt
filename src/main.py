import streamlit as st

from src.opt.record import Record
from src.opt.solve import solve


def main() -> None:
    # st.title("Hello, Streamlit!")

    records = [[Record(0, 0, 0) for _ in range(7)] for _ in range(6)]

    solve(records, [0, 2, 1])


if __name__ == "__main__":
    main()
