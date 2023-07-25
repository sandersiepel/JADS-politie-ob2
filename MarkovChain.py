import pandas as pd
import networkx as nx
from pyvis.network import Network


class MarkovChain:
    def __init__(self, df):
        self.df = df
        self.main()

    def main(self):
        self.validate_input()  # Make sure that the input df is valid.
        matrix = self.make_transition_matrix()
        graph = self.make_graph(matrix)
        self.save_graph(graph)

    def validate_input(self):
        # Check if self.df has the right columns
        return None

    def make_transition_matrix(self):
        transitions = self.df.location.values.tolist()

        matrix = pd.crosstab(
            pd.Series(transitions[:-1], name="from"),
            pd.Series(transitions[1:], name="to"),
            normalize=0,
        )

        return matrix

    def make_graph(self, matrix):
        graph = nx.DiGraph()

        # Add nodes to the graph
        for source, targets in matrix.T.items():
            graph.add_node(source)
            for target, weight in targets.items():
                if weight > 0:
                    graph.add_node(target)
                    graph.add_edge(
                        source, target, weight=weight, title=round(weight, 3)
                    )

        print(f"Edges: {len(graph.edges)}, nodes: {len(graph.nodes)}")

        return graph

    def save_graph(self, graph):
        net = Network(height="750px", width="100%", directed=True)
        net.from_nx(graph)
        # net.show_buttons()

        net.set_options(
            """
            const options = {
                "nodes": {
                    "borderWidth": null,
                    "borderWidthSelected": null,
                    "opacity": 1,
                    "font": {
                    "size": 8,
                    "strokeWidth": 1
                    },
                    "size": null
                },
                "edges": {
                    "arrows": {
                    "to": {
                        "enabled": true,
                        "scaleFactor": 0.2
                    }
                    },
                    "color": {
                    "inherit": true
                    },
                    "hoverWidth": 1,
                    "selectionWidth": 1,
                    "selfReferenceSize": null,
                    "selfReference": {
                    "angle": 0.7853981633974483
                    },
                    "smooth": {
                    "forceDirection": "none"
                    }
                },
                "physics": {
                    "minVelocity": 0.75
                }
            }
        """
        )

        net.write_html("output/markovchain.html", notebook=False)
