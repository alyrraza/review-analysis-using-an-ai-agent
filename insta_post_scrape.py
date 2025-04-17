from apify_client import ApifyClient
from dotenv import load_dotenv
import os
import pandas as pd

class InstagramPostScrapper:
    def __init__(self, username: str, results_limit: int = 30, api_token: str = None):
        load_dotenv()
        
        # If API token is not provided, fetch from environment variables
        self.api_token = api_token or os.getenv("APIFY_API_TOKEN")
        
        if not self.api_token:
            raise ValueError("API Token is required and not found in environment variables or constructor.")
        
        self.username = username
        self.results_limit = results_limit
        
        # Initialize the ApifyClient with the API token
        self.client = ApifyClient(self.api_token)
        
        # Input for the Apify Actor
        self.run_input = {
            "username": [self.username],
            "resultsLimit": self.results_limit
        }

        # DataFrame to store post data
        self.df = pd.DataFrame(columns=["url", "commentsCount", "likesCount"])
        self.avg_likes = 0
        self.avg_comments = 0

    def run_scraper(self):
        """Run the Instagram post scraper Actor."""
        run = self.client.actor("nH2AHrwxeTRJoN5hX").call(run_input=self.run_input)
        return run["defaultDatasetId"]

    def fetch_data(self, dataset_id: str):
        """Fetch and filter the post data."""
        post_data = []
        for item in self.client.dataset(dataset_id).iterate_items():
            filtered = {
                "url": item.get("url"),
                "commentsCount": item.get("commentsCount", 0),
                "likesCount": item.get("likesCount", 0)
            }
            post_data.append(filtered)

        # Convert to DataFrame and assign to self.df
        self.df = pd.DataFrame(post_data)

    def calculate_averages(self):
        """Calculate average likes and comments from the DataFrame."""
        if not self.df.empty:
            self.avg_likes = self.df["likesCount"].mean()
            self.avg_comments = self.df["commentsCount"].mean()

    def get_data(self):
        """Getter function to return the DataFrame containing fetched data."""
        return self.df

    def get_avg_likes(self):
        """Getter function to return the average likes."""
        return self.avg_likes

    def get_avg_comments(self):
        """Getter function to return the average comments."""
        return self.avg_comments

    def run(self):
        """Run the scraper, fetch data, and calculate averages."""
        dataset_id = self.run_scraper()
        self.fetch_data(dataset_id)

        # Calculate averages
        self.calculate_averages()

        return self.avg_likes, self.avg_comments
