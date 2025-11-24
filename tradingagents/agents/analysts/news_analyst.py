from langchain_core.messages import SystemMessage, HumanMessage
from datetime import datetime, timedelta
from tradingagents.agents.utils.agent_utils import get_news, get_global_news
from tradingagents.dataflows.config import get_config


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        # Compute 7-day lookback window
        try:
            end_dt = datetime.strptime(current_date, "%Y-%m-%d")
            start_dt = end_dt - timedelta(days=7)
            start_date = start_dt.strftime("%Y-%m-%d")
        except Exception:
            # Fallback: use the same day if parsing fails
            start_date = current_date

        # Fetch company-specific news and global macro news via tools
        company_news = ""
        global_news = ""
        try:
            company_news = get_news.invoke({"ticker": ticker, "start_date": start_date, "end_date": current_date}) or ""
        except Exception:
            company_news = ""
        try:
            global_news = get_global_news.invoke({"curr_date": current_date, "look_back_days": 7, "limit": 8}) or ""
        except Exception:
            global_news = ""

        # Build a data-grounded instruction and feed fetched data to the LLM
        system_instruction = (
            "You are a news researcher tasked with analyzing recent news and trends over the past week. "
            "Write a comprehensive, data-grounded report relevant for trading and macroeconomics. "
            "Use the provided fetched news data as primary evidence. "
            "Do not simply state that trends are mixed. Provide detailed and nuanced insights with implications. "
            "Append a concise Markdown table at the end summarizing key points.\n\n"
            f"Context:\n"
            f"- Current date: {current_date}\n"
            f"- Company: {ticker}\n\n"
            f"Fetched company news ({ticker}, {start_date} to {current_date}):\n{company_news}\n\n"
            f"Fetched global/macro news (last 7 days):\n{global_news}\n"
        )

        messages = [
            SystemMessage(content=system_instruction),
            HumanMessage(content=f"Produce the final report for {ticker} using the fetched data above."),
        ]
        result = llm.invoke(messages)

        report = ""

        # Use the generated content as the report
        report = getattr(result, "content", "") or ""

        return {
            "messages": [result],
            "news_report": report,
        }

    return news_analyst_node
