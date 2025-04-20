import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import seaborn as sns
import numpy as np
import pandas as pd

class Visualizer:
    def __init__(self, plot_dir: str = "data/plots"):
        self.plot_dir = plot_dir
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

    def plot_probabilities(self, probabilities: dict, filename: str, threshold: float = 0.5) -> str:
        """Tạo biểu đồ cột tĩnh cho xác suất của các nhãn bằng matplotlib."""
        labels = list(probabilities.keys())
        probs = list(probabilities.values())

        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, probs, color="skyblue")

        plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
        plt.xlabel("Labels", fontsize=12)
        plt.ylabel("Probability", fontsize=12)
        plt.title("KNN Classification Probabilities", fontsize=14)
        plt.ylim(0, 1)
        plt.xticks(rotation=90, ha='center')
        plt.legend()

        plot_path = os.path.join(self.plot_dir, filename)
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        return plot_path

    def plot_probabilities_interactive(self, probabilities: dict, title: str = "KNN Classification Probabilities", threshold: float = 0.5) -> str:
        """Tạo biểu đồ cột tương tác bằng plotly và trả về HTML."""
        labels = list(probabilities.keys())
        probs = list(probabilities.values())

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=labels,
                y=probs,
                name="Probabilities",
                marker_color='skyblue'
            )
        )

        fig.add_shape(
            type="line",
            x0=0,
            x1=len(labels)-1,
            y0=threshold,
            y1=threshold,
            line=dict(color="red", dash="dash"),
            name=f"Threshold ({threshold})"
        )

        fig.update_layout(
            title=title,
            xaxis_title="Labels",
            yaxis_title="Probability",
            xaxis_tickangle=45,
            showlegend=True,
            height=600,
            width=1000
        )

        return fig.to_html(full_html=False)

    def plot_top_k_probabilities(self, probabilities: dict, filename: str, k: int = 5, threshold: float = 0.5, title: str = None) -> str:
        """Tạo biểu đồ cột tĩnh cho top-k nhãn có xác suất cao nhất."""
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:k]
        labels, probs = zip(*sorted_probs)
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(labels, probs, color="skyblue")
        plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
        plt.xlabel("Labels")
        plt.ylabel("Probability")
        plt.title(title if title else f"Top-{k} KNN Classification Probabilities")  # Sử dụng tiêu đề tùy chỉnh
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha='center')
        plt.legend()
        plot_path = os.path.join(self.plot_dir, filename)
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        return plot_path

    def plot_prob_distribution(self, prob_dicts: list, filename: str) -> str:
        """Tạo biểu đồ box plot để hiển thị phân bố xác suất của các nhãn cho từng ảnh."""
        prob_values = [list(prob_dict.values()) for prob_dict in prob_dicts]
        
        plt.figure(figsize=(10, 6))
        plt.boxplot(prob_values, vert=False, patch_artist=True)
        plt.xlabel("Probability")
        plt.ylabel("Images")
        plt.title("Distribution of KNN Classification Probabilities per Image")
        plt.yticks(ticks=range(1, len(prob_dicts) + 1), labels=[f"Image {i+1}" for i in range(len(prob_dicts))])
        plot_path = os.path.join(self.plot_dir, filename)
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        return plot_path

    def plot_heatmap(self, prob_dicts: list, labels: list, filename: str, top_k: int = 20) -> str:
        """Tạo heatmap để so sánh xác suất của nhiều ảnh, chỉ hiển thị top-k nhãn."""
        # Tính xác suất trung bình cho mỗi nhãn
        avg_probs = {label: np.mean([prob_dict.get(label, 0) for prob_dict in prob_dicts]) for label in labels}
        # Sắp xếp và lấy top-k nhãn
        top_labels = sorted(avg_probs, key=avg_probs.get, reverse=True)[:top_k]
        
        prob_matrix = []
        for prob_dict in prob_dicts:
            prob_matrix.append([prob_dict.get(label, 0) for label in top_labels])
        
        df = pd.DataFrame(prob_matrix, columns=top_labels, index=[f"Image {i+1}" for i in range(len(prob_dicts))])

        plt.figure(figsize=(12, 8))
        sns.heatmap(df, annot=True, cmap="Blues", vmin=0, vmax=1)
        plt.title("Heatmap of KNN Classification Probabilities (Top Labels)")
        plt.xlabel("Labels")
        plt.ylabel("Images")

        plot_path = os.path.join(self.plot_dir, filename)
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        return plot_path

    def plot_heatmap_interactive(self, prob_dicts: list, labels: list, top_k: int = 20) -> str:
        """Tạo heatmap tương tác bằng plotly, chỉ hiển thị top-k nhãn."""
        # Tính xác suất trung bình cho mỗi nhãn
        avg_probs = {label: np.mean([prob_dict.get(label, 0) for prob_dict in prob_dicts]) for label in labels}
        top_labels = sorted(avg_probs, key=avg_probs.get, reverse=True)[:top_k]
        
        prob_matrix = []
        for prob_dict in prob_dicts:
            prob_matrix.append([prob_dict.get(label, 0) for label in top_labels])

        fig = go.Figure(data=go.Heatmap(
            z=prob_matrix,
            x=top_labels,
            y=[f"Image {i+1}" for i in range(len(prob_dicts))],
            colorscale="Blues",
            zmin=0,
            zmax=1,
            text=[[f"{val:.2f}" for val in row] for row in prob_matrix],
            texttemplate="%{text}",
            hoverinfo="z"
        ))

        fig.update_layout(
            title="Heatmap of KNN Classification Probabilities (Top Labels)",
            xaxis_title="Labels",
            yaxis_title="Images",
            xaxis_tickangle=45,
            height=600,
            width=1000
        )

        return fig.to_html(full_html=False)

    def plot_loss(self, loss_history: list, filename: str, title: str = "NMF Loss Over Iterations") -> str:
        """Tạo biểu đồ đường tĩnh cho hàm lỗi bằng matplotlib."""
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history, marker='o', color='blue', label='Loss')
        plt.xlabel("Iteration", fontsize=12)
        plt.ylabel("Loss (Frobenius Norm)", fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True)
        plt.legend()

        plot_path = os.path.join(self.plot_dir, filename)
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        return plot_path

    def plot_loss_interactive(self, loss_history: list, title: str = "NMF Loss Over Iterations") -> str:
        """Tạo biểu đồ đường tương tác cho hàm lỗi bằng plotly."""
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(range(len(loss_history))),
                y=loss_history,
                mode='lines+markers',
                name='Loss',
                line=dict(color='blue')
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Iteration",
            yaxis_title="Loss (Frobenius Norm)",
            showlegend=True,
            height=600,
            width=1000
        )

        return fig.to_html(full_html=False)
    
    def plot_grouped_violin(self, prob_matrix: list, labels: list, filename: str, title: str = "Probability Distribution") -> str:
        """Tạo biểu đồ violin plot để hiển thị phân bố xác suất của top-k nhãn cho một nhóm ảnh."""
        df = pd.DataFrame(prob_matrix, columns=labels)
        
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=df, orient="v", palette="Blues")
        plt.xlabel("Labels")
        plt.ylabel("Probability")
        plt.title(title)
        plt.xticks(rotation=45, ha='center')
        plot_path = os.path.join(self.plot_dir, filename)
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        return plot_path