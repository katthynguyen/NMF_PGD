# api/routes.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from core.nmf import NMF
from core.image_processor import ImageProcessor
from core.knn_classifier import KNNClassifier
from core.visualizer import Visualizer
import os
import logging
import uuid
import config.constants as Const
from typing import List, Dict, Any
from collections import defaultdict
import numpy as np

router = APIRouter()
logger = logging.getLogger(__name__)


# Khởi tạo các thành phần
image_processor = ImageProcessor(target_size=tuple(Const.IMAGE_SIZE))
nmf = NMF(
    n_components=Const.N_FEATURES,
    learning_rate=Const.LEARNING_RATE,
    max_iter=Const.MAX_ITER,
)
knn = KNNClassifier(n_neighbors=Const.KNN_NEIGHBORS)
visualizer = Visualizer()
is_trained = False


async def training(
    train_image_paths: List[str], n_components: int = 10, n_neighbors: int = 5
):
    """Huấn luyện mô hình NMF và KNN dựa trên tập ảnh và nhãn."""
    global is_trained, nmf, knn

    # Tiền xử lý tập ảnh huấn luyện
    image_train, train_labels = image_processor.process_load_image(train_image_paths)
    image_train = image_train[:200]
    train_labels = train_labels[:200]
    V_train = image_processor.preprocess_image(image_train)

    if V_train is None:
        raise HTTPException(
            status_code=400, detail="Không thể xử lý tập ảnh huấn luyện."
        )

    try:
        # Khởi tạo và huấn luyện NMF
        nmf.n_components = n_components
        loss_history = nmf.fit(V_train)  # Lấy lịch sử lỗi
        print(loss_history)

        # Trích xuất đặc trưng H cho huấn luyện KNN
        H_train = nmf.transform(V_train)
        knn.n_neighbors = n_neighbors
        knn.fit(H_train.T, train_labels)

        is_trained = True
        logger.info("Huấn luyện mô hình hoàn tất.")

        return loss_history  # Trả về lịch sử lỗi

    except Exception as e:
        logger.error(f"Lỗi khi huấn luyện mô hình: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi huấn luyện: {str(e)}")


async def action_extract_features(
    image_paths: List[str], n_components: int = 10, n_neighbors: int = 5
) -> Dict[str, Any]:
    """Nhận tập ảnh, trích xuất đặc trưng, phân loại, và trả về kết quả cùng biểu đồ."""
    global nmf, knn
    # Tiền xử lý toàn bộ tập ảnh
    V_new = image_processor.preprocess_image(image_paths[:210])
    if V_new is None:
        raise HTTPException(status_code=400, detail="Không thể xử lý tập ảnh.")

    try:
        # Trích xuất đặc trưng bằng NMF
        H_new = nmf.transform(V_new)
        features = H_new.T.tolist()

        # Phân loại bằng KNN cho từng ảnh
        predicted_labels, probabilities = knn.predict(H_new.T)
        prob_dicts = [knn.get_label_probabilities(prob) for prob in probabilities]

        # Tạo biểu đồ top-k cho từng ảnh
        top_k_plot_paths = []
        top_k_plot_htmls = []
        for i, prob_dict in enumerate(prob_dicts):
            top_k_filename = f"top_k_{uuid.uuid4()}.png"
            image_filename = os.path.basename(image_paths[i])
            predicted_label = predicted_labels[i]
            plot_title = f"Top-5 KNN Probabilities for {image_filename} (Predicted: {predicted_label})"
            
            top_k_path = visualizer.plot_top_k_probabilities(
                prob_dict, top_k_filename, k=5, threshold=0.5, title=plot_title
            )
            top_k_html = visualizer.plot_probabilities_interactive(
                prob_dict,
                title=plot_title,
            )
            top_k_plot_paths.append(top_k_path)
            top_k_plot_htmls.append(top_k_html)
    
        grouped_probs = defaultdict(list)
        for i, (label, prob_dict) in enumerate(zip(predicted_labels, prob_dicts)):
            grouped_probs[label].append((image_paths[i], prob_dict))
        
        # Tạo biểu đồ tổng hợp cho từng nhóm nhãn dự đoán
        grouped_plot_paths = {}
        for label, images_probs in grouped_probs.items():
            # Lấy danh sách prob_dict cho nhóm này
            group_prob_dicts = [prob_dict for _, prob_dict in images_probs]
            # Lấy top-k nhãn dựa trên xác suất trung bình trong nhóm
            all_labels = set().union(*[set(prob_dict.keys()) for prob_dict in group_prob_dicts])
            avg_probs = {lbl: np.mean([prob_dict.get(lbl, 0) for prob_dict in group_prob_dicts]) for lbl in all_labels}
            top_k_labels = sorted(avg_probs, key=avg_probs.get, reverse=True)[:5]
            
            # Tạo ma trận xác suất cho top-k nhãn
            prob_matrix = []
            for prob_dict in group_prob_dicts:
                prob_matrix.append([prob_dict.get(lbl, 0) for lbl in top_k_labels])
            
            # Tạo violin plot cho nhóm
            group_filename = f"group_{label}_{uuid.uuid4()}.png"
            group_plot_path = visualizer.plot_grouped_violin(
                prob_matrix, top_k_labels, group_filename, title=f"Probability Distribution for Images Predicted as {label}"
            )
            grouped_plot_paths[label] = group_plot_path
                
        
        # Tạo biểu đồ phân bố xác suất cho tất cả ảnh
        dist_filename = f"dist_{uuid.uuid4()}.png"
        dist_plot_path = visualizer.plot_prob_distribution(prob_dicts, dist_filename)

        # Tạo heatmap tổng hợp với top-k nhãn
        all_labels = sorted(
            set().union(*[set(prob_dict.keys()) for prob_dict in prob_dicts])
        )
        heatmap_filename = f"heatmap_{uuid.uuid4()}.png"
        heatmap_path = visualizer.plot_heatmap(
            prob_dicts, all_labels, heatmap_filename, top_k=20
        )
        heatmap_html = visualizer.plot_heatmap_interactive(
            prob_dicts, all_labels, top_k=20
        )

        # Tạo biểu đồ hàm lỗi
        loss_filename = f"loss_{uuid.uuid4()}.png"
        loss_plot_path = visualizer.plot_loss(
            nmf.loss_history,
            loss_filename,
            title=f"NMF Loss Over Iterations (Components={n_components})",
        )
        loss_html = visualizer.plot_loss_interactive(
            nmf.loss_history,
            title=f"NMF Loss Over Iterations (Components={n_components})",
        )

        # Tổng hợp kết quả
        results = []
        for i in range(len(image_paths[:20])):
            results.append(
                {
                    "image_path": image_paths[i],
                    "features": features[i],
                    "predicted_label": predicted_labels[i],
                    "probabilities": prob_dicts[i],
                    "top_k_plot_path": top_k_plot_paths[
                        i
                    ],  # Thay plot_path bằng top_k_plot_path
                    "top_k_plot_html": top_k_plot_htmls[
                        i
                    ],  # Thay plot_html bằng top_k_plot_html
                }
            )

        return {
            "results": results,
            "heatmap_path": heatmap_path,
            "heatmap_html": heatmap_html,
            "dist_plot_path": dist_plot_path,  # Thêm biểu đồ phân bố xác suất
            "loss_plot_path": loss_plot_path,
            "loss_html": loss_html,
            "n_components": n_components,
            "n_neighbors": n_neighbors,
        }

    except Exception as e:
        logger.error(f"Lỗi khi trích xuất đặc trưng: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract-features")
async def extract_features():
    """Nhận ảnh, trích xuất đặc trưng, phân loại, và trả về kết quả cùng biểu đồ."""
    image_train, train_labels = image_processor.process_load_image(Const.FOLDER_TRAIN)
    image_train = image_train[:200]
    train_labels = train_labels[:200]
    V_train = image_processor.preprocess_image(image_train)
    await training(Const.FOLDER_TRAIN)
    if not is_trained:
        raise HTTPException(status_code=503, detail="Mô hình chưa được huấn luyện.")
    # # Lưu ảnh tạm thời
    temp_path, lables = image_processor.process_load_image(Const.FOLDER_IMAGES)
    res = await action_extract_features(temp_path)

    return JSONResponse(content="done")


@router.get("/dist-plot/{dist_filename}")
async def get_dist_plot(dist_filename: str):
    """Trả về file biểu đồ phân bố xác suất."""
    dist_plot_path = os.path.join("data/plots", dist_filename)
    if not os.path.exists(dist_plot_path):
        raise HTTPException(status_code=404, detail="Biểu đồ phân bố không tồn tại.")
    return FileResponse(dist_plot_path, media_type="image/png")


@router.get("/plot-html/{index}")
async def get_plot_html(index: int):
    """Trả về biểu đồ tương tác dưới dạng HTML."""
    temp_path, _ = image_processor.process_load_image(Const.FOLDER_IMAGES)
    res = await action_extract_features(temp_path, n_components=10, n_neighbors=5)
    if index < 0 or index >= len(res["results"]):
        raise HTTPException(status_code=404, detail="Biểu đồ không tồn tại.")
    return HTMLResponse(content=res["results"][index]["top_k_plot_html"])  # Updated key

@router.get("/grouped-plot/{label}")
async def get_grouped_plot(label: str):
    """Trả về file biểu đồ nhóm theo nhãn dự đoán."""
    temp_path, _ = image_processor.process_load_image(Const.FOLDER_IMAGES)
    res = await action_extract_features(temp_path, n_components=10, n_neighbors=5)
    grouped_plot_path = res["grouped_plot_paths"].get(label)
    if not grouped_plot_path or not os.path.exists(grouped_plot_path):
        raise HTTPException(status_code=404, detail="Biểu đồ nhóm không tồn tại.")
    return FileResponse(grouped_plot_path, media_type="image/png")