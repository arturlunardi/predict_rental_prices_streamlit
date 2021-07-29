import streamlit as st
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import pandas as pd


@st.cache
def plot_histogram(data, x, nbins, height, width, margin, title_text=None):
    fig = px.histogram(data, x=x, nbins=nbins)
    fig.update_layout(bargap=0.05, height=height, width=width, title_text=title_text, margin=dict(t=margin,
                                                                                                  b=margin
                                                                                                )
    )
    return fig


@st.cache
def plot_scatter(data, x, y, height, width, margin, residual=False, title_text=None):
    if residual:
        fig = px.scatter(
        data, x=x, y=y,
        marginal_x='histogram', marginal_y='histogram',
        color='split', trendline='ols', opacity=.5
        )

        # add an annotation with the train R2
        fig.add_annotation(
                    xref="paper",
                    yref="paper",
                    x=0.73,
                    y=0.98,
                    text=f"Train R²: {round(r2_score(data.loc[data['split'] == 'train']['rent amount (R$)'], data.loc[data['split'] == 'train']['prediction']), 3)}",
                    bordercolor="#c7c7c7",
                    borderwidth=2,
                    borderpad=4,
                    bgcolor="red",
                    opacity=0.8,
                    showarrow=False,
                    font=dict(
                        family="Courier New, monospace",
                        size=12,
                        color="#ffffff"
                        )
                    )

        # add an annotation with the test R2
        fig.add_annotation(
                    xref="paper",
                    yref="paper",
                    x=0.57,
                    y=0.98,
                    text=f"Test R²: {round(r2_score(data.loc[data['split'] == 'test']['rent amount (R$)'], data.loc[data['split'] == 'test']['prediction']), 3)}",
                    bordercolor="#c7c7c7",
                    borderwidth=2,
                    borderpad=4,
                    bgcolor="blue",
                    opacity=0.8,
                    showarrow=False,
                    font=dict(
                        family="Courier New, monospace",
                        size=12,
                        color="#ffffff"
                        )
                    )

        # add an annotation with the train RMSE
        fig.add_annotation(
                    xref="paper",
                    yref="paper",
                    x=0.73,
                    y=0.89,
                    text=f"Train RMSE: {round(mean_squared_error(data.loc[data['split'] == 'train']['rent amount (R$)'], data.loc[data['split'] == 'train']['prediction'], squared=False), 2)}",
                    bordercolor="#c7c7c7",
                    borderwidth=2,
                    borderpad=4,
                    bgcolor="red",
                    opacity=0.8,
                    showarrow=False,
                    font=dict(
                        family="Courier New, monospace",
                        size=12,
                        color="#ffffff"
                        )
                    )

        # add an annotation with the test RMSE
        fig.add_annotation(
                    xref="paper",
                    yref="paper",
                    x=0.56,
                    y=0.89,
                    text=f"Test RMSE: {round(mean_squared_error(data.loc[data['split'] == 'test']['rent amount (R$)'], data.loc[data['split'] == 'test']['prediction'], squared=False), 2)}",
                    bordercolor="#c7c7c7",
                    borderwidth=2,
                    borderpad=4,
                    bgcolor="blue",
                    opacity=0.8,
                    showarrow=False,
                    font=dict(
                        family="Courier New, monospace",
                        size=12,
                        color="#ffffff"
                        )
                    )


    else:
        fig = px.scatter(data, x=x, y=y)

    fig.update_layout(bargap=0.05, height=height, width=width, title_text=title_text, margin=dict(t=margin,
                                                                                                  b=margin
                                                                                                )
    )
    return fig


@st.cache(hash_funcs={pd.DataFrame: lambda x: x})
def plot_boxplot(data, x, y, height, width, margin, color=None, single_box=False, model_name=None, custom_feature=None, custom_target=None, title_text=None):
    if single_box:
        fig = go.Figure(
        go.Box(
            y = data.loc[(data['name'] == model_name) & (data['custom_features'] == custom_feature) & (data['custom_target'] == custom_target)]['all_scores_cv'].iloc[0],
            name = model_name,
            marker_color='darkblue',
            boxpoints='all',
            jitter=0.3,
            boxmean=True
            )
        )
    else:
        fig = px.box(data, x=x, y=y, color=color)

    fig.update_layout(bargap=0.05, height=height, width=width, title_text=title_text, margin=dict(t=margin,
                                                                                                  b=margin
                                                                                                )
    )
    return fig


@st.cache
def plot_countplot(data, x, height, width, margin, title_text=None):
    fig = px.histogram(data, x=x, color=x)
    fig.update_layout(bargap=0.05, height=height, width=width, title_text=title_text, margin=dict(t=margin,
                                                                                                  b=margin
                                                                                                )
    )
    return fig


@st.cache
def plot_heatmap(corr_matrix, height, margin, title_text=None):
    fig = go.Figure(
        go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.index.values,
        y=corr_matrix.columns.values,
        colorscale='RdBu_R',
        zmax=1,
        zmin=-1
        )
    )

    fig.update_layout(bargap=0.05, height=height, width=700, title_text=title_text, margin=dict(t=margin,
                                                                                                  b=margin
                                                                                                )
    )

    return fig


@st.cache
def plot_distplot(y_real, y_predict, height, width, margin, title_text=None):
    fig = ff.create_distplot(
    [y_real, y_predict],
    ['Real', 'Predicted'],
    bin_size=150,
    # show_hist=False
    )

    fig.update_layout(bargap=0.05, height=height, width=width, title_text=title_text, margin=dict(t=margin,
                                                                                                  b=margin
                                                                                                )
    )

    return fig


@st.cache
def plot_bar(data, x, y, height, width, margin, title_text=None):
    fig = px.bar(data, x=x, y=y, color=x)

    fig.update_layout(bargap=0.05, height=height, width=width, title_text=title_text, margin=dict(t=margin,
                                                                                                  b=margin
                                                                                                )
    )

    return fig