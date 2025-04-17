
from insta_post_scrape import InstagramPostScrapper
from insta_comment_scrape import InstaCommentScrapper

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class InstagramSentimentAnalyzer:
    def __init__(self, username):
        self.username = username
        self.posts_df = None
        self.comments_df = None
        self.label_map = {
            'LABEL_0': 'Negative',
            'LABEL_1': 'Neutral',
            'LABEL_2': 'Positive'
        }

        # Load Model and Tokenizer
        model_path = "D:/ML/LLm/models--microsoft--Phi-3-mini-4k-instruct/snapshots/Review Analysis using AI agent/Fine tuned model"
        tokenizer = AutoTokenizer.from_pretrained("callmesan/indic-bert-roman-urdu-fine-grained")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1)

        # Set Seaborn style
        sns.set(style="whitegrid")
        plt.rcParams.update({
            "figure.figsize": (10, 6),
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12
        })

    def run_analysis(self):
        self._scrape_posts()
        self._scrape_comments()
        self._analyze_sentiment()

    def _scrape_posts(self):
        post_scraper = InstagramPostScrapper(username=self.username)
        post_scraper.run()
        self.avg_Comments = post_scraper.get_avg_comments()
        self.avg_Likes = post_scraper.get_avg_likes()
        post_scraper.get_avg_likes()
        self.posts_df = post_scraper.get_data()
        self.post_urls = self.posts_df["url"].tolist()

    def _scrape_comments(self):
        comment_scraper = InstaCommentScrapper()
        for url in self.post_urls:
            comment_scraper.scrape_comments(post_url=url)
        self.comments_df = comment_scraper.get_comments_df()

    def _analyze_sentiment(self):
        sentiments = []
        for comment in self.comments_df["commentText"]:
            try:
                result = self.classifier(comment)[0]
                label = self.label_map[result["label"]]
                score = result["score"]
                sentiments.append(f"{label} ({score:.2f})")
            except Exception:
                sentiments.append("Error")
        self.comments_df["Sentiment"] = sentiments
        self.comments_df["sentiment"] = self.comments_df["Sentiment"].apply(lambda x: x.split(" ")[0] if x != "Error" else "Error")

    # ---------------------- Visualization Methods ----------------------
    def get_comments(self):
        return self.avg_Comments
    def get_likes(self):
        return self.avg_Likes
    
    def plot_sentiment_distribution(self):
        fig, ax = plt.subplots()
        sentiment_counts = self.comments_df['sentiment'].value_counts()
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='pastel', ax=ax)
        ax.set_title("Sentiment Distribution of Comments")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Number of Comments")
        return fig

    def plot_sentiment_over_time(self):
        if 'timestamp' in self.comments_df.columns:
            fig, ax = plt.subplots()
            df_time = self.comments_df.copy()
            df_time['timestamp'] = pd.to_datetime(df_time['timestamp'])
            df_time.set_index('timestamp', inplace=True)
            sentiment_over_time = df_time.resample('W')['sentiment'].value_counts().unstack().fillna(0)
            sentiment_over_time.plot(marker='o', ax=ax)
            ax.set_title("Sentiment Trend Over Time")
            ax.set_xlabel("Time")
            ax.set_ylabel("Number of Comments")
            return fig

    def plot_likes_vs_sentiment(self):
        fig, ax = plt.subplots()
        sns.boxplot(data=self.comments_df, x='sentiment', y='likesCount', palette='Set2', ax=ax)
        ax.set_title("Likes per Comment by Sentiment")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Likes on Comment")
        return fig

    def plot_replies_vs_sentiment(self):
        fig, ax = plt.subplots()
        sns.boxplot(data=self.comments_df, x='sentiment', y='repliesCount', palette='Set1', ax=ax)
        ax.set_title("Replies per Comment by Sentiment")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Replies on Comment")
        return fig

    def plot_comment_count_by_sentiment(self):
        fig, ax = plt.subplots()
        sns.countplot(data=self.comments_df, x='sentiment', palette='coolwarm', ax=ax)
        ax.set_title("Comment Count by Sentiment")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Number of Comments")
        return fig

    def plot_engagement_overview(self):
        fig, ax = plt.subplots()
        engagement_summary = self.comments_df[['likesCount', 'repliesCount']].mean()
        sns.barplot(x=engagement_summary.index, y=engagement_summary.values, palette='viridis', ax=ax)
        ax.set_title("Average Engagement on Comments")
        ax.set_ylabel("Average Count")
        ax.set_xlabel("Engagement Type")
        return fig

    def plot_sentiment_engagement_heatmap(self):
        self.comments_df['likesCount'] = pd.to_numeric(self.comments_df['likesCount'], errors='coerce')
        self.comments_df['repliesCount'] = pd.to_numeric(self.comments_df['repliesCount'], errors='coerce')
        fig, ax = plt.subplots()
        heat_data = self.comments_df.groupby('sentiment')[['likesCount', 'repliesCount']].mean()
        sns.heatmap(heat_data, annot=True, cmap='YlGnBu', ax=ax)
        ax.set_title("Average Likes & Replies per Sentiment")
        return fig

    def plot_likes_distribution(self):
        fig, ax = plt.subplots()
        sns.histplot(self.comments_df['likesCount'], kde=True, bins=20, color='skyblue', ax=ax)
        ax.set_title("Distribution of Likes on Comments")
        ax.set_xlabel("Likes Count")
        ax.set_ylabel("Frequency")
        return fig

    def plot_replies_distribution(self):
        fig, ax = plt.subplots()
        sns.histplot(self.comments_df['repliesCount'], kde=True, bins=20, color='lightcoral', ax=ax)
        ax.set_title("Distribution of Replies on Comments")
        ax.set_xlabel("Replies Count")
        ax.set_ylabel("Frequency")
        return fig

    def plot_engagement_weighted_sentiment(self):
        fig, ax = plt.subplots()
        weighted_sentiment = self.comments_df.groupby('sentiment')[['likesCount', 'repliesCount']].sum()
        weighted_sentiment.plot(kind='bar', stacked=True, colormap='Accent', ax=ax)
        ax.set_title("Engagement-Weighted Sentiment")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Total Likes + Replies")
        return fig

    def get_top_engaging_comments(self):
        top_liked = self.comments_df.sort_values(by='likesCount', ascending=False).head(5)
        top_replied = self.comments_df.sort_values(by='repliesCount', ascending=False).head(5)
        weighted_summary = self.comments_df.groupby('sentiment')[['likesCount', 'repliesCount']].sum().reset_index()
        return top_liked, top_replied, weighted_summary
