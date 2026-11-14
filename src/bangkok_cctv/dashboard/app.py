"""Streamlit landing page for the Bangkok CCTV course project."""

import streamlit as st


def main() -> None:
    """Display project scope and implementation status."""
    st.set_page_config(page_title="Bangkok CCTV Image Analysis", layout="wide")
    st.title("Bangkok CCTV Image Analysis")
    st.write(
        "A university Data Science and Data Engineering project exploring "
        "Bangkok traffic through CCTV images and external data."
    )
    st.info("Project scaffold ready. Data collection and analysis are planned.")
    st.subheader("Planned workflow")
    st.write(
        "CCTV images ? vehicle counts ? validated traffic records ? "
        "external data integration ? analysis ? interactive dashboard"
    )


if __name__ == "__main__":
    main()
