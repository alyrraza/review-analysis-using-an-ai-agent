import streamlit as st
from analyzer import InstagramSentimentAnalyzer
import matplotlib.pyplot as plt

# Title
st.title("📈 Instagram Sentiment Insights Dashboard")

# Input bar for username
username = st.text_input("Enter the Instagram username:")

# Analyze button
if st.button("Analyze") and username:
    with st.spinner("Fetching posts and analyzing comments..."):
        try:
            # Initialize analyzer and run sentiment analysis
            analyzer = InstagramSentimentAnalyzer(username)
            analyzer.run_analysis()

            # Fetch average metrics
            avg_likes = analyzer.get_likes()
            avg_comments = analyzer.get_comments()

            # Display key engagement metrics
            st.subheader("🔹 Engagement Metrics Overview")
            col1, col2 = st.columns(2)
            col1.metric("Average Likes per Post", f"{avg_likes:.2f}")
            col2.metric("Average Comments per Post", f"{avg_comments:.2f}")

            st.divider()

            # Sentiment distribution across all comments
            st.subheader("💬 Overall Sentiment Distribution")
            st.write("This chart shows the proportion of positive, negative, and neutral comments across the user's recent posts.")
            st.pyplot(analyzer.plot_sentiment_distribution())

            # Sentiment trend over time
            st.subheader("📅 Sentiment Trend Over Time")
            st.write("Tracks how comment sentiments have changed over time, providing insight into audience mood shifts.")
            fig_time = analyzer.plot_sentiment_over_time()
            if fig_time:
                st.pyplot(fig_time)
            else:
                st.info("Unable to generate trend chart — timestamps may be missing or malformed.")

            # Likes vs sentiment
            st.subheader("❤️ Average Likes by Sentiment")
            st.write("Visualizes how posts associated with different sentiment categories perform in terms of likes.")
            st.pyplot(analyzer.plot_likes_vs_sentiment())

            # Replies vs sentiment
            st.subheader("💬 Average Replies by Sentiment")
            st.write("Highlights how comment sentiment correlates with audience engagement through replies.")
            st.pyplot(analyzer.plot_replies_vs_sentiment())

            # Count of comments per sentiment category
            st.subheader("🧮 Comment Count by Sentiment")
            st.write("Displays the number of comments that fall into each sentiment category.")
            st.pyplot(analyzer.plot_comment_count_by_sentiment())

            # Likes and Replies overview
            st.subheader("📊 Engagement Overview")
            st.write("Summarizes post performance in terms of likes and replies across the analyzed dataset.")
            st.pyplot(analyzer.plot_engagement_overview())

            # Heatmap of sentiment and engagement
            st.subheader("🔥 Sentiment vs Engagement Heatmap")
            st.write("A heatmap that shows the interaction between comment sentiment and overall engagement levels.")
            st.pyplot(analyzer.plot_sentiment_engagement_heatmap())

            # Distribution of likes
            st.subheader("📈 Likes Distribution Across Comments")
            st.write("Shows how likes are distributed across individual comments.")
            st.pyplot(analyzer.plot_likes_distribution())

            # Distribution of replies
            st.subheader("📈 Replies Distribution Across Comments")
            st.write("Shows how replies are distributed across individual comments.")
            st.pyplot(analyzer.plot_replies_distribution())

            # Weighted sentiment
            st.subheader("📊 Engagement-Weighted Sentiment")
            st.write("This chart weighs sentiment scores by the engagement levels (likes/replies) of comments to reflect impact.")
            st.pyplot(analyzer.plot_engagement_weighted_sentiment())

            # Display top engaging comments
            st.subheader("🏆 Most Engaging Comments")
            top_liked, top_replied, _ = analyzer.get_top_engaging_comments()

            st.write("Here are the top 5 most liked and replied comments based on sentiment and engagement:")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Top 5 Most Liked Comments**")
                st.dataframe(top_liked[["commentText", "likesCount", "repliesCount", "sentiment"]])

            with col2:
                st.markdown("**Top 5 Most Replied Comments**")
                st.dataframe(top_replied[["commentText", "likesCount", "repliesCount", "sentiment"]])

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
