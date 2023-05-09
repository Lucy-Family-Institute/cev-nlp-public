from wordcloud import WordCloud
import cv2  # FIXME: use another library to load an image, opencv is too heavy
import os
from PIL import Image
import plotly.graph_objects as go
import plotly
import networkx as nx
import random

current_path = os.path.dirname(os.path.abspath(__file__))


def mostRelevantData(df, topn=35):
    """
    # TODO: Juan Pablo -> documentation
    """
    df = df.sort_values("value", ascending=False)
    dat = df.to_numpy()
    shortData = dat[:topn]
    shortDataReverse = shortData[::-1]
    return shortData, shortDataReverse


def random_color_func(
    word=None,
    font_size=None,
    position=None,
    orientation=None,
    font_path=None,
    random_state=None,
):
    """
    # TODO: Juan Pablo -> documentation
    """
    h = int(360.0 * 0.0 / 255.0)
    s = int(100.0 * 255.0 / 255.0)
    l = int(100.0 * float(random_state.randint(60, 120)) / 255.0)

    return "hsl({}, {}%, {}%)".format(h, s, l)


def frequency_graph(df, wordcloud_path: str, file_name: str):
    """
    # TODO: Juan Pablo -> documentation
    """
    dat = mostRelevantData(df)
    dictWordCloud = {}
    for i in range(dat[0].shape[0]):
        dictWordCloud[dat[0][i][0]] = dat[0][i][1]

    wc = WordCloud(
        background_color="white",
        width=1000,
        height=500,
        max_words=100,  # len(dictWordCloud),
        color_func=random_color_func,
    )

    wc.generate_from_frequencies(dictWordCloud)
    wc.to_file(wordcloud_path)
    image1 = cv2.imread(wordcloud_path)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image1 = Image.fromarray(image1)

    trace1 = go.Bar(
        y=list(dat[1][0:, 0]),
        x=list(dat[1][0:, 1]),
        orientation="h",
        # hoverinfo="name",
        marker_color="rgba(50, 171, 96, 1.0)",
        width=0.9,
    )

    trace2 = go.Scatter(x=[], y=[], xaxis="x2", yaxis="y2")
    font = dict(family="Arial", size=12, color="rgb(42, 74, 130)")

    data = [trace1, trace2]
    layout = go.Layout(
        images=list(
            [
                dict(
                    source=image1,
                    xref="paper",
                    yref="paper",
                    x=1.05,
                    y=0.3,
                    sizex=0.4,
                    sizey=0.9,
                    xanchor="right",
                    yanchor="bottom",
                )
            ]
        ),
        title="<b>Palabras más usadas</b>",
        yaxis=dict(
            range=[(len(dat[1]) - 10), len(dat[1])],
            title="<b>Palabras</b>",
        ),
        xaxis=dict(
            domain=[0.1, 0.5],
            range=[0, (dat[0][0, 1] + 50)],
            title="<b>Frecuencia</b>",
            showline=False,
            visible=True,
        ),
        xaxis2=dict(
            title="<b>Palabra</b>",
            domain=[0.6, 1],
            visible=False,
        ),
        yaxis2=dict(
            anchor="x2",
            visible=True,
            showline=True,
            showticklabels=False,
            linecolor="rgba(102, 102, 102, 0.8)",
            linewidth=2,
            domain=[0, 1],
        ),
        font=font,
        showlegend=False,
        paper_bgcolor="rgb(248, 248, 255)",
        plot_bgcolor="rgb(248, 248, 255)",
    )

    fig = dict(data=data, layout=layout)
    plotly.offline.plot(fig, filename=file_name)


def heatMap(valx, valy, y_label, x_label, matrix, file_name):
    """
    # TODO: Juan Pablo -> documentation
    """
    relaciones = valx
    grup = valy
    matrix = matrix

    trace = go.Heatmap(
        x=relaciones, y=grup, z=matrix, type="heatmap", colorscale="Blues"
    )

    layout = go.Layout(
        title="<b>Matriz de coocurrencias</b>",
        yaxis=dict(
            range=[0, len(valy)],
            title=f"<b> {y_label} </b>",
        ),
        xaxis=dict(
            range=[0, len(valx)],
            showline=False,
            visible=True,
            title=f"<b> {x_label} </b>",
        ),
        paper_bgcolor="rgb(248, 248, 255)",
        plot_bgcolor="rgb(248, 248, 255)",
    )
    data = [trace]
    fig = dict(data=data, layout=layout)
    plotly.offline.plot(fig, filename=file_name)


def net(graph, type1, type2, file_name):
    """
    # TODO: Juan Pablo -> documentation
    """
    ##Pos
    nx.spring_layout(graph)
    spring_3D = nx.spring_layout(graph, dim=3, seed=18)
    # we need to seperate the X,Y,Z coordinates for Plotly
    x_nodes = [spring_3D(node)[0] for node in graph.nodes()]  # x-coordinates of nodes
    y_nodes = [spring_3D(node)[1] for node in graph.nodes()]  # y-coordinates of nodes
    z_nodes = [spring_3D(node)[2] for node in graph.nodes()]  # z-coordinates of nodes
    # we  need to create lists that contain the starting and ending coordinates of each edge.
    # TODO: Juan Pablo -> set individual widths for each edge. information is in graph[edges[0]][edges[1]]["weight"]
    x_edges = [
        [spring_3D[u][0], spring_3D[v][0], None]
        for u, v, data in graph.edges(data=True)
    ]
    y_edges = [[spring_3D[u][1], spring_3D[v][1], None] for u, v in graph.edges()]
    z_edges = [[spring_3D[u][2], spring_3D[v][2], None] for u, v in graph.edges()]

    trace_edges = go.Scatter3d(
        x=x_edges,
        y=y_edges,
        z=z_edges,
        mode="lines",
        line=dict(color="black"),
        hoverinfo="none",
    )
    # create a trace for the nodes
    # TODO: set colours of each node with respect to their property "type". Juan Pablo, how to set the colour of each node individually?
    # graph.nodes(data=True)[node]["type"] -> organisation/person/verb/etc. one colour for each type.
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    trace_nodes = go.Scatter3d(
        x=x_nodes,
        y=y_nodes,
        z=z_nodes,
        mode="markers",
        marker=dict(
            symbol="circle",
            size=10,
            color="rgb({},{},{})".format(
                r, g, b
            ),  # color the nodes according to their community
            colorscale=["lightgreen", "magenta"],  # either green or mageneta
            line=dict(color="black", width=0.5),
        ),  # TODO: Juan Pablo, how to set size of each node individually? graph.nodes(data=True)[node]["size"] is the size
        text=[str(node) for node in graph.nodes()],
        hoverinfo="text",
    )
    # trace_MrHi = go.Scatter3d(x=[x_nodes[0]],
    #                 y=[y_nodes[0]],
    #                 z=[z_nodes[0]],
    #                 mode='markers',
    #                 name='Ana',
    #                 marker=dict(symbol='circle',
    #                             size=15,
    #                             color=('rgb({},{},{})').format(10,g,b),
    #                             line=dict(color='black', width=0.6)
    #                             ),
    #                 text = club_labels_2[0],
    #                 hoverinfo = 'text')

    # we need to set the axis for the plot
    axis = dict(
        showbackground=False,
        showline=False,
        zeroline=False,
        showgrid=False,
        showticklabels=False,
        title="",
    )
    # also need to create the layout for our plot
    layout = go.Layout(
        title=f"Network - {type1} vs {type2}",
        # width=650,
        # height=625,
        showlegend=False,
        scene=dict(
            xaxis=dict(axis),
            yaxis=dict(axis),
            zaxis=dict(axis),
        ),
        margin=dict(t=150),
        hovermode="closest",
    )
    # Include the traces we want to plot and create a figure
    data = [trace_edges, trace_nodes]
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename=file_name)
