import numpy as np
import cv2

pointMapper = [
    # contorno del volto
    (127, 1),                           #  0
    (234, 1),                           #  1
    (93,  1),                           #  2
    (132, 1),                           #  3
    (58,  1),                           #  4
    (172, 1),                           #  5
    (150, 1),                           #  6
    (176, 1),                           #  7
    (152, 1),                           #  8
    (400, 1),                           #  9
    (379, 1),                           # 10
    (397, 1),                           # 11
    (288, 1),                           # 12
    (361, 1),                           # 13
    (323, 1),                           # 14
    (454, 1),                           # 15
    (356, 1),                           # 16
    
    # sopracciglio destro
    (70,   46, 1, 1),                   # 17
    (63,   53, 1, 1),                   # 18
    (105,  52, 1, 1),                   # 19
    (66,   65, 1, 1),                   # 20
    (107,  55, 1, 1),                   # 21
    
    # sopracciglio sinistro
    (336, 285, 1, 1),                   # 22
    (296, 295, 1, 1),                   # 23
    (334, 282, 1, 1),                   # 24
    (293, 283, 1, 1),                   # 25
    (300, 276, 1, 1),                   # 26

    # naso
    (168,   6, 1, 1),                   # 27
    (197, 1),                           # 28
    (195,   5, 1, 3),                   # 29
    (4,     1, 2, 1),                   # 30
    #
    ( 98, 240, 1, 1),                   # 31
    ( 98, 164, 1, 1),                   # 32
    (164,   2, 2, 1),                   # 33
    (327, 164, 1, 1),                   # 34
    (327, 460, 1, 1),                   # 35

    # occhio destro
    (33,  1),                           # 36
    (160, 1),                           # 37
    (158, 1),                           # 38
    (133, 1),                           # 39
    (153, 1),                           # 40
    (144, 1),                           # 41

    # occhio sinistro
    (362, 1),                           # 42
    (385, 1),                           # 43
    (387, 1),                           # 44
    (263, 1),                           # 45
    (373, 1),                           # 46
    (380, 1),                           # 47

    # bocca esterna
    (61,  1),                           # 48
    (40,  1),                           # 49
    (37,  39,  1, 1),                   # 50
    (0,   1),                           # 51
    (267, 269, 1, 1),                   # 52
    (270, 1),                           # 53
    (291, 1),                           # 54
    (321, 1),                           # 55
    (314, 405, 1, 1),                   # 56
    (17,  1),                           # 57
    (84,  181, 1, 1),                   # 58
    (91,  1),                           # 59

    # bocca interna
    (78,  1),                           # 60
    (81,  1),                           # 61
    (13,  1),                           # 62
    (311, 1),                           # 63
    (308, 1),                           # 64
    (402, 1),                           # 65
    (14,  1),                           # 66
    (178, 1)                            # 67
]


tris = np.array(
    [
        # contorno del volto
        [127, 234, 227], # 1
        [234, 177, 123], # 2
        [ 93,  58, 147], # 3
        [132, 172, 213], # 4
        [ 58, 136, 192], # 5
        [172, 150, 135], # 6
        [136, 149, 210], # 7
        [149, 148,  32], # 8
        [148, 377, 175], # 0
        [377, 378, 262], # 8
        [365, 430, 378], # 7
        [397, 364, 379], # 6
        [288, 416, 365], # 5
        [361, 433, 397], # 4
        [323, 376, 288], # 3
        [454, 352, 401], # 2
        [356, 447, 454], # 1

        # sopracciglio destro
        [ 70,  46,  63], # 1
        [ 63,  53, 105], # 2
        [105,  52,  66], # 3
        [ 66,  65, 107], # 4
        [107,  55,   9], # 5

        # sopracciglio sinistro
        [336,   9, 285], # 5
        [296, 336, 295], # 4
        [334, 296, 282], # 3
        [293, 334, 283], # 2
        [300, 293, 276], # 1

        # naso
        [168, 122, 351], # 0
        [  6, 196, 419], # 0
        [195,  51, 281], # 0
        [  4,  44, 274], # 0
        #
        #[102, 115, 198], # 1
        [ 64, 218, 131], # 1
        [ 79, 242, 241], # 2  --> ok
        [141, 370,  19], # 0  --> ok
        [309, 461, 462], # 2  --> ok
        [294, 360, 438]  # 1
        #[331, 420, 344]  # 1
    ]
).astype(int)

FILTER_POINTS = True # False # 
FAKE_VIS = np.ones(468).astype(bool)

class SmartCamLandmarker():
    def __init__(self, calib=None):
        self.visible_dc = None
        self.thermal_dc = None
        self.visible_cm = None
        self.thermal_cm = None
        self.visible_to_thermal_rmat = None
        self.visible_to_thermal_tvec = None
        if calib is not None:
            self.set_calib(calib)


    def get_calib(self):
        return self.calib


    def set_calib(self, calib):
        if calib.shape[0]==3 or calib.shape[0]==4:
            self.visible_cm = calib[:3]
        elif calib.shape[0]==8 or calib.shape[0]==9:
            self.visible_cm = calib[:3]
            self.thermal_cm = calib[3:6]
            self.visible_to_thermal_rmat = cv2.Rodrigues(calib[6])[0].astype(np.float64)
            self.visible_to_thermal_tvec = calib[7] * 100  # mediapipe uses centimeters
        else:
            raise NotImplementedError
        self.calib = calib

    def extract_points(self, geom):
        visible_points, vVis = self.extract_rgb_points(geom)
        thermal_points, tVis = self.extract_ir_points(geom)
        return visible_points, vVis, thermal_points, tVis

    def extract_rgb_points(self, geom):
        pose_to_visible_rvec, pose_to_visible_tvec = geom[0], geom[1]
        vertices = geom[2:]

        pose_to_visible_rmat = cv2.Rodrigues(pose_to_visible_rvec)[0].astype(np.float64)

        ij_visible = cv2.projectPoints(
            vertices[:, None],
            pose_to_visible_rmat,
            pose_to_visible_tvec,
            self.visible_cm,
            self.visible_dc,
        )[0][:, 0]
        visible_points = self.remap_points(ij_visible)
        vVis = self.compute_visibility(vertices, pose_to_visible_rmat, pose_to_visible_tvec, 
                                       self.visible_cm, self.visible_dc)
        return visible_points.astype(np.float32), vVis

    def extract_ir_points(self, geom):
        pose_to_visible_rvec, pose_to_visible_tvec = geom[0], geom[1]
        vertices = geom[2:]

        rmat = self.visible_to_thermal_rmat
        tvec = self.visible_to_thermal_tvec
            
        pose_to_visible_rmat = cv2.Rodrigues(pose_to_visible_rvec)[0].astype(np.float64)
        pose_to_thermal_rmat = rmat @ pose_to_visible_rmat
        pose_to_thermal_tvec = (tvec + (rmat @ pose_to_visible_tvec[:, None])[:, 0])

        ij_thermal = cv2.projectPoints(
            vertices[:, None],
            pose_to_thermal_rmat,
            pose_to_thermal_tvec,
            self.thermal_cm,
            self.thermal_dc,
        )[0][:, 0]
        thermal_points = self.remap_points(ij_thermal)
        tVis = self.compute_visibility(vertices, pose_to_thermal_rmat, pose_to_thermal_tvec, 
                                       self.thermal_cm, self.thermal_dc)
        return thermal_points.astype(np.float32), tVis

    def remap_points(self, mediapoints):
        if not FILTER_POINTS:
            return mediapoints

        points = np.empty((0, 2))
        for tp in pointMapper:
            length = len(tp) // 2
            ps = tp[:length]
            ws = tp[-length:]
            p = np.zeros(2)
            for s, w in zip(ps, ws):
                p += mediapoints[s]*w
            p /= sum(ws)
            points = np.vstack((points, p))
        return points


    def compute_visibility(self, verts, rvec, tvec, cm, dc):

        if not FILTER_POINTS:
            return FAKE_VIS[:verts.shape[0]]
        
        verts_tris = verts[tris]
        ij_quads = cv2.projectPoints(verts_tris.reshape(-1, 1, 3), rvec, tvec, cm, dc)[0].reshape(-1, 3, 2)

        visibility = []
        for ijs in ij_quads:
            ax, ay = ijs[1] - ijs[0]
            bx, by = ijs[2] - ijs[1]
            visibility.append(ax * by - ay * bx <= 0)

        while len(visibility) < 68:
            visibility.append(True)
        visibility[8] = True
        return np.array(visibility).astype(bool)
