import cv2
import numpy as np



class MarkerDetector:
    def __init__(self, init_frame: np.ndarray, marker_width: float = 0.10) -> None:
        self.marker_width = marker_width
        self.height, self.width = init_frame.shape[:2]

        self.dict_aruco_target = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.dict_aruco_nav = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
        self.parameters = cv2.aruco.DetectorParameters_create()

        # Atributos auxiliares
        self.loaded = False
        self.matrix = None
        self.distortion = None
        self.new_camera_matrix = None
        self.roi = None
        self.x_offset = self.y_offset = None

        self.load_calibration()

    def load_calibration(self):
        if not self.loaded:
            with np.load("calibration_files/camcalib.npz") as X:
                self.matrix = X["mtx"]
                self.distortion = X["dist"]

            # Calcula New Camera Matrix, Offsets e novos Height e Width
            self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
                self.matrix,
                self.distortion,
                (self.width, self.height),
                1,
                (self.width, self.height),
            )
            self.x_offset, self.y_offset, self.width, self.height = self.roi
            self.loaded = True

    def detect_markers(self, gray_frame: np.ndarray, marker_dict: dict):
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray_frame, marker_dict, parameters=self.parameters
        )
        return np.array(corners), ids

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        undistorted_frame = cv2.undistort(
            frame, self.matrix, self.distortion, None, self.new_camera_matrix
        )
        # cropped_frame = undistorted_frame[
        #     self.y_offset : self.y_offset + self.height,
        #     self.x_offset : self.x_offset + self.width,
        # ]
        gray_frame = cv2.cvtColor(undistorted_frame, cv2.COLOR_RGB2GRAY)

        # Detect markers on the preprocessed gray_frame
        target_corners, target_ids = self.detect_markers(
            gray_frame, self.dict_aruco_target
        )
        nav_corners, nav_ids = self.detect_markers(gray_frame, self.dict_aruco_nav)

        # Draw markers and estimate pose on the original frame
        self.draw_markers(frame, target_corners, target_ids, "Target")
        self.draw_markers(frame, nav_corners, nav_ids, "Navigation")
        distancia = self.estimate_pose_and_draw_axes(frame, target_corners, target_ids)
        self.estimate_pose_and_draw_axes(frame, nav_corners, nav_ids)

        target_distance = self.estimate_pose_and_draw_axes(frame, target_corners, target_ids, "Target")
        nav_distance = self.estimate_pose_and_draw_axes(frame, nav_corners, nav_ids, "Navigation")

        return frame, target_distance, nav_distance

    def draw_markers(
        self,
        frame: np.ndarray,
        corners: np.ndarray,
        ids: list,
        marker_class: str = "",
    ):
        if ids is None or marker_class is None:
            return

        # Add offset to corners
        # corners[:, :, :, [0, 1]] += [self.x_offset, self.y_offset]
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Add text labels indicating the marker class
        for i, id_value in enumerate(ids):
            marker_label = f"{marker_class} ID {id_value}"

            x, y = int(corners[i][0][:, 0].mean()), int(corners[i][0][:, 1].mean())
            cv2.putText(
                frame,
                marker_label,
                (x, y + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
    def estimate_post(self,frame: np.ndarray,corners: np.array,ids: list,):
        if ids is None:
            return
        rotations_vecs, translation_vecs, _ = cv2.aruco.estimatePoesSingleMarkers(
            corners, self.marker_width, self.matrix, self.distortion
        )
        pass





    def estimate_pose_and_draw_axes(self,frame: np.ndarray,corners: np.array,ids: list,):

        if ids is None:
            return

        rotations_vecs, translation_vecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.marker_width, self.matrix, self.distortion
        )
        corners[:, :, :, [0, 1]] += [self.x_offset, self.y_offset]
        for corner, rot_vec, tran_vec in zip(corners, rotations_vecs, translation_vecs):
            center_x = corner[0][:, 0].mean()
            center_y = corner[0][:, 1].mean()

            z_string = f"{tran_vec[0][2]:.2f}"
            cv2.drawFrameAxes(
                frame, self.matrix, self.distortion, rot_vec, tran_vec, 0.1
            )
            cv2.putText(
                frame,
                f"Distance {z_string} m",
                (int(center_x), int(center_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )
        return z_string

# Inicializa a câmera
cap = cv2.VideoCapture(0)

# Lê um frame inicial para inicializar o MarkerDetector
ret, init_frame = cap.read()
if not ret:
    print("Falha ao obter o frame inicial.")
    exit()

# Inicializa o detector de marcadores
marker_detector = MarkerDetector(init_frame, marker_width=0.18)

while True:
    # Captura um frame da câmera
    ret, frame = cap.read()

    if not ret:
        print("Falha ao capturar o frame. Saindo...")
        break

    # Processa o frame para detectar e desenhar os marcadores
    processed_frame, distancia = marker_detector.process_frame(frame)

    # Exibe o frame processado
    cv2.imshow("Marcadores ArUco", processed_frame)
    print(distancia)
    # Encerra o programa se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos e fecha as janelas
cap.release()
cv2.destroyAllWindows()
