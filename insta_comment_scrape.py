from apify_client import ApifyClient
from dotenv import load_dotenv
import os
import pandas as pd

class InstaCommentScrapper:
    def __init__(self, api_token: str = None):
        """Initialize the comment scrapper with API token from .env or constructor."""
        load_dotenv()

        self.api_token = api_token or os.getenv("APIFY_API_TOKEN")
        if not self.api_token:
            raise ValueError("API Token is required and not found in environment variables or constructor.")

        self.client = ApifyClient(self.api_token)

        # DataFrame to store all comments with extended fields
        self.df = pd.DataFrame(columns=["postUrl", "commentText", "likesCount", "repliesCount", "timestamp"])

    def scrape_comments(self, post_url: str, results_limit: int = 15):
        """Scrapes comments from a single Instagram post and appends to the DataFrame."""
        run_input = {
            "directUrls": [post_url],
            "resultsLimit": results_limit
        }

        # Run the Apify actor and fetch comments
        run = self.client.actor("SbK00X0JYCPblD2wp").call(run_input=run_input)
        comments = []

        for item in self.client.dataset(run["defaultDatasetId"]).iterate_items():
            if "text" in item:
                comments.append({
                    "postUrl": post_url,
                    "commentText": item.get("text"),
                    "likesCount": item.get("likesCount", 0),
                    "repliesCount": item.get("repliesCount", 0),
                    "timestamp": item.get("timestamp", None)
                })

        # Append new comments to the main DataFrame
        new_comments_df = pd.DataFrame(comments)
        self.df = pd.concat([self.df, new_comments_df], ignore_index=True)

    def get_comments_df(self):
        """Returns the DataFrame containing all scraped comments."""
        return self.df
