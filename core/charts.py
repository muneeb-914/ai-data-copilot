import pandas as pd
import plotly.express as px

def render_chart(df: pd.DataFrame, chart_spec: dict):
    chart_type = chart_spec.get("chart_type")
    x = chart_spec.get("x")
    y = chart_spec.get("y")
    color = chart_spec.get("color")
    title = chart_spec.get("title", "")

    # Validate columns exist
    for col in [x, y, color]:
        if col and col not in df.columns:
            return None

    try:
        if chart_type == "histogram":
            return px.histogram(df, x=x, color=color, title=title)

        elif chart_type == "bar":
            if y:
                grouped = df.groupby(x)[y].mean().reset_index()
                return px.bar(grouped, x=x, y=y, title=title)
            else:
                counts = df[x].value_counts().reset_index()
                counts.columns = [x, "count"]
                return px.bar(counts, x=x, y="count", title=title)

        elif chart_type == "boxplot":
            return px.box(df, x=color, y=x, title=title) if color else px.box(df, y=x, title=title)

        elif chart_type == "scatter":
            if x and y:
                return px.scatter(df, x=x, y=y, color=color, title=title)

        elif chart_type == "heatmap":
            numeric_df = df.select_dtypes(include='number')
            corr = numeric_df.corr()
            return px.imshow(corr, text_auto=True, title=title)

        elif chart_type == "pie":
            counts = df[x].value_counts().reset_index()
            counts.columns = [x, "count"]
            return px.pie(counts, names=x, values="count", title=title)

    except Exception:
        return None