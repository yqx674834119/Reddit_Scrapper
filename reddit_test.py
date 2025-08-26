import praw

# 填入你的 Reddit API 配置
# Reddit API Credentials (https://www.reddit.com/prefs/apps)
REDDIT_CLIENT_ID='_UtweeChG9QBLuom1n8tpw'
REDDIT_CLIENT_SECRET='yqsipo4HtR1XSS2QDDA-QBPGtYo6kg'
REDDIT_USER_AGENT = 'market:v1.0 (by /u/Think-Hearing8673)'
REDDIT_USERNAME='Think-Hearing8673'
REDDIT_PASSWORD='bGzJXVT!67u9gzD'

# OpenAI API Key (https://platform.openai.com/api-keys)
OPENAI_API_KEY='2dcaeb9d-1b48-4fa5-afcb-e3a5f957bc55'
ARK_API_KEY='2dcaeb9d-1b48-4fa5-afcb-e3a5f957bc55'

# 初始化 Reddit 实例
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT,
    username=REDDIT_USERNAME,
    password=REDDIT_PASSWORD,
)

# 测试访问 Reddit 首页
try:
    print("✅ 成功登录 Reddit！下面是 r/popular 的前 5 条帖子：")
    for submission in reddit.subreddit("popular").hot(limit=5):
        print(f"- {submission.title} (👍 {submission.score})")
except Exception as e:
    print("❌ 出现错误：")
    print(e)
