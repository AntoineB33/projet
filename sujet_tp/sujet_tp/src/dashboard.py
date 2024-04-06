import argparse
import subprocess

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the clustering dashboard with specified data paths.")
    parser.add_argument("--path_data", type=str, required=True, help="Path to the image data and analysis results.")
    parser.add_argument("--path_images", type=str, required=True, help="Path to the images folder")
    args = parser.parse_args()

    # Construct the Streamlit command
    streamlit_command = [
        "streamlit", "run", "dashboard_clustering.py",  # rt.py should be the name of your actual Streamlit script file
        "--", "--path_data", args.path_data, "--path_images", args.path_images
    ]

    # Execute the Streamlit app
    subprocess.run(streamlit_command)

if __name__ == "__main__":
    main()
