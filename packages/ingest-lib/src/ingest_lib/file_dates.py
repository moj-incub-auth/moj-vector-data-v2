import logging
import subprocess
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class GitFileDates:
    """
    A class to retrieve the last modification dates of files in a git repository.
    """

    def __init__(self, repo_path: str = "."):
        self.repo_path = repo_path

    def get_file_dates(self, filter_pattern: Optional[str] = None) -> Dict[str, str]:

        file_dates = {}

        # Get all files tracked by git in HEAD
        try:
            result = subprocess.run(
                ["git", "ls-tree", "-r", "--name-only", "HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )

            files = result.stdout.strip().split("\n")

            # For each file, get the last commit date
            for filename in files:
                if not filename:
                    continue

                # Apply filter if provided
                if filter_pattern and filter_pattern not in filename:
                    continue

                # Get the last commit date for this file
                date_result = subprocess.run(
                    ["git", "log", "-1", "--format=%ad", "--", filename],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True,
                    check=True,
                )

                commit_date = date_result.stdout.strip()
                if commit_date:
                    file_dates[filename] = commit_date

            return file_dates

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Git command failed: {e.stderr}")
        except Exception as e:
            raise RuntimeError(f"Error retrieving file dates: {str(e)}")


# Example usage
if __name__ == "__main__":
    git_dates = GitFileDates()

    # Get all files with their dates
    all_files = git_dates.get_file_dates()
    logger.debug("All files:")
    if logger.isEnabledFor(logging.DEBUG):
        for file, date in all_files.items():
            logger.debug(f"{date} {file}")

        logger.debug("\n" + "=" * 80 + "\n")

    # Get only README.md.njk files (equivalent to the grep in the bash script)
    readme_files = git_dates.get_file_dates(filter_pattern="README.md.njk")
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("README.md.njk files:")
        for file, date in readme_files.items():
            logger.debug(f"{date} {file}")
