import numpy as np

BOSON_HEADER = b"boson320"
LEPTON2_HEADER = b"lepton2"
LEPTON3_HEADER = b"lepton3"
A6x5_HEADER = b"a6x5"
A400_HEADER = b"a400"
MERC_HEADER = b"mer_mono"
CROP_HEADER = b"venus_crop"
DAL16_HEADER = b"dal640_16"
ROTATE_HEADER = b"venus_rotate"
HEADER_MAX_SIZE = 32
GEOM_HEADER = b"face_geom"
PTS_HEADER = b"face_pts"

# zsc: "compressed" common format (two cams)
frameZSC_dtype = np.dtype(
    [
        ("hdr1", np.uint8, HEADER_MAX_SIZE),
        ("t1", np.uint64),
        ("ser1", np.uint8, HEADER_MAX_SIZE),
        ("n1", np.uint64),
        ("hdr2", np.uint8, HEADER_MAX_SIZE),
        ("t2", np.uint64),
        ("ser2", np.uint8, HEADER_MAX_SIZE),
        ("n2", np.uint64),
        ("ghdr", np.uint8, HEADER_MAX_SIZE),
        ("gt", np.uint64),
        ("geom", np.float32, (470, 3)),
        ("calib", np.float32, (9, 3)),
        ("score", np.int64),
    ]
)

# msc: half "compressed" common format (1 visible cam, points with mediapipe)
frameMSC_dtype = np.dtype(
    [
        ("hdr2", np.uint8, HEADER_MAX_SIZE),
        ("t2", np.uint64),
        ("ser2", np.uint8, HEADER_MAX_SIZE),
        ("n2", np.uint64),
        ("ghdr", np.uint8, HEADER_MAX_SIZE),
        ("gt", np.uint64),
        ("geom", np.float32, (470, 3)),
        ("calib", np.float32, (3, 3)),
        ("score", np.int64),
    ]
)

# dsc: half "compressed" common format (1 thermal cam, points with dlib)
frameDSC_dtype = np.dtype(
    [
        ("hdr1", np.uint8, HEADER_MAX_SIZE),
        ("t1", np.uint64),
        ("ser1", np.uint8, HEADER_MAX_SIZE),
        ("n1", np.uint64),
        ("phdr", np.uint8, HEADER_MAX_SIZE),
        ("pt", np.uint64),
        ("points", np.float32, (68, 2)),
        ("calib", np.float32, (3, 3)),
        ("score", np.int64),
    ]
)


# venus (cropped and resized) + boson
frameVTB_dtype = np.dtype(
    [
        ("hdr1", np.uint8, HEADER_MAX_SIZE),
        ("t1", np.uint64),
        ("ser1", np.uint8, HEADER_MAX_SIZE),
        ("n1", np.uint64),
        ("img1", np.uint8, (450, 550)),
        ("hdr2", np.uint8, HEADER_MAX_SIZE),
        ("t2", np.uint64),
        ("ser2", np.uint8, HEADER_MAX_SIZE),
        ("n2", np.uint64),
        ("img2", np.uint16, (256, 320)),
        ("ghdr", np.uint8, HEADER_MAX_SIZE),
        ("gt", np.uint64),
        ("geom", np.float32, (470, 3)),
        ("calib", np.float32, (9, 3)),
        ("score", np.int64),
    ]
)


# a615/a655 (thermal only)
frameA6x5_dtype = np.dtype(
    [
        ("hdr1", np.uint8, HEADER_MAX_SIZE),
        ("t1", np.uint64),
        ("ser1", np.uint8, HEADER_MAX_SIZE),
        ("n1", np.uint64),
        ("img1", np.uint16, (480, 640)),
        ("phdr", np.uint8, HEADER_MAX_SIZE),
        ("pt", np.uint64),
        ("points", np.float32, (68, 2)),
        ("calib", np.float32, (3, 3)),
        ("score", np.int64),
    ]
)

# a615/a655 + mercury (cropped)
frameADM_dtype = np.dtype(
    [
        ("hdr1", np.uint8, HEADER_MAX_SIZE),
        ("t1", np.uint64),
        ("ser1", np.uint8, HEADER_MAX_SIZE),
        ("n1", np.uint64),
        ("img1", np.uint16, (480, 640)),
        ("hdr2", np.uint8, HEADER_MAX_SIZE),
        ("t2", np.uint64),
        ("ser2", np.uint8, HEADER_MAX_SIZE),
        ("n2", np.uint64),
        ("img2", np.uint8, (1024, 1024)),
        ("ghdr", np.uint8, HEADER_MAX_SIZE),
        ("gt", np.uint64),
        ("geom", np.float32, (470, 3)),
        ("calib", np.float32, (9, 3)),
        ("score", np.int64),
    ]
)

# device alab (shutterless 16 bit)
frameDal16_dtype = np.dtype(
    [
        ("hdr", np.uint8, HEADER_MAX_SIZE),
        ("t", np.uint64),
        ("ser", np.uint8, HEADER_MAX_SIZE),
        ("n", np.uint64),
        ("img", np.uint16, (480, 640)),
        ("phdr", np.uint8, HEADER_MAX_SIZE),
        ("pt", np.uint64),
        ("points", np.int32, (68, 2)),
        ("calib", np.float32, (3, 3)),
        ("score", np.int64),
    ]
)

# boson
frameBoson_dtype = np.dtype(
    [
        ("hdr", np.uint8, HEADER_MAX_SIZE),
        ("t", np.uint64),
        ("ser", np.uint8, HEADER_MAX_SIZE),
        ("n", np.uint64),
        ("img", np.uint16, (256, 320)),
        ("phdr", np.uint8, HEADER_MAX_SIZE),
        ("pt", np.uint64),
        ("points", np.float32, (68, 2)),
        ("calib", np.float32, (3, 3)),
        ("score", np.int64),
    ]
)

# boson + venus (rotated and resized)
frameBosonRotate_dtype = np.dtype(
    [
        ("hdr1", np.uint8, HEADER_MAX_SIZE),
        ("t1", np.uint64),
        ("ser1", np.uint8, HEADER_MAX_SIZE),
        ("n1", np.uint64),
        ("img1", np.uint16, (256, 320)),
        ("hdr2", np.uint8, HEADER_MAX_SIZE),
        ("t2", np.uint64),
        ("ser2", np.uint8, HEADER_MAX_SIZE),
        ("n2", np.uint64),
        ("img2", np.uint8, (720, 540)),
        ("ghdr", np.uint8, HEADER_MAX_SIZE),
        ("gt", np.uint64),
        ("geom", np.float32, (470, 3)),
        ("calib", np.float32, (9, 3)),
        ("score", np.int64),
    ]
)

# lepton 2 (80 x 60)
frameLepton2_dtype = np.dtype(
    [
        ("hdr", np.uint8, HEADER_MAX_SIZE),
        ("t", np.uint64),
        ("ser", np.uint8, HEADER_MAX_SIZE),
        ("n", np.uint64),
        ("img", np.uint16, (60, 80)),
        ("phdr", np.uint8, HEADER_MAX_SIZE),
        ("pt", np.uint64),
        ("points", np.float32, (68, 2)),
        ("calib", np.float32, (3, 3)),
        ("score", np.int64),
    ]
)

# lepton 2 + venus (rotated and resized)
frameL2Rotate_dtype = np.dtype(
    [
        ("hdr1", np.uint8, HEADER_MAX_SIZE),
        ("t1", np.uint64),
        ("ser1", np.uint8, HEADER_MAX_SIZE),
        ("n1", np.uint64),
        ("img1", np.uint16, (60, 80)),
        ("hdr2", np.uint8, HEADER_MAX_SIZE),
        ("t2", np.uint64),
        ("ser2", np.uint8, HEADER_MAX_SIZE),
        ("n2", np.uint64),
        ("img2", np.uint8, (720, 540)),
        ("ghdr", np.uint8, HEADER_MAX_SIZE),
        ("gt", np.uint64),
        ("geom", np.float32, (470, 3)),
        ("calib", np.float32, (9, 3)),
        ("score", np.int64),
    ]
)

# lepton 3 (160 x 120)
frameLepton3_dtype = np.dtype(
    [
        ("hdr", np.uint8, HEADER_MAX_SIZE),
        ("t", np.uint64),
        ("ser", np.uint8, HEADER_MAX_SIZE),
        ("n", np.uint64),
        ("img", np.uint16, (120, 160)),
        ("phdr", np.uint8, HEADER_MAX_SIZE),
        ("pt", np.uint64),
        ("points", np.float32, (68, 2)),
        ("calib", np.float32, (3, 3)),
        ("score", np.int64),
    ]
)

# lepton 3 + venus (rotated and resized)
frameL3Rotate_dtype = np.dtype(
    [
        ("hdr1", np.uint8, HEADER_MAX_SIZE),
        ("t1", np.uint64),
        ("ser1", np.uint8, HEADER_MAX_SIZE),
        ("n1", np.uint64),
        ("img1", np.uint16, (120, 160)),
        ("hdr2", np.uint8, HEADER_MAX_SIZE),
        ("t2", np.uint64),
        ("ser2", np.uint8, HEADER_MAX_SIZE),
        ("n2", np.uint64),
        ("img2", np.uint8, (720, 540)),
        ("ghdr", np.uint8, HEADER_MAX_SIZE),
        ("gt", np.uint64),
        ("geom", np.float32, (470, 3)),
        ("calib", np.float32, (9, 3)),
        ("score", np.int64),
    ]
)

# a400 (thermal only)
frameA400_dtype = np.dtype(
    [
        ("hdr1", np.uint8, HEADER_MAX_SIZE),
        ("t1", np.uint64),
        ("ser1", np.uint8, HEADER_MAX_SIZE),
        ("n1", np.uint64),
        ("img1", np.uint16, (240, 320)),
        ("phdr", np.uint8, HEADER_MAX_SIZE),
        ("pt", np.uint64),
        ("points", np.float32, (68, 2)),
        ("calib", np.float32, (3, 3)),
        ("score", np.int64),
    ]
)

# a400 + mercury (cropped and resized)
frameA4M_dtype = np.dtype(
    [
        ("hdr1", np.uint8, HEADER_MAX_SIZE),
        ("t1", np.uint64),
        ("ser1", np.uint8, HEADER_MAX_SIZE),
        ("n1", np.uint64),
        ("img1", np.uint16, (240, 320)),
        ("hdr2", np.uint8, HEADER_MAX_SIZE),
        ("t2", np.uint64),
        ("ser2", np.uint8, HEADER_MAX_SIZE),
        ("n2", np.uint64),
        ("img2", np.uint8, (512, 512)),
        ("ghdr", np.uint8, HEADER_MAX_SIZE),
        ("gt", np.uint64),
        ("geom", np.float32, (470, 3)),
        ("calib", np.float32, (9, 3)),
        ("score", np.int64),
    ]
)

# mercury (cropped and resized)
frameMerc_dtype = np.dtype(
    [
        ("hdr", np.uint8, HEADER_MAX_SIZE),
        ("t", np.uint64),
        ("ser", np.uint8, HEADER_MAX_SIZE),
        ("n", np.uint64),
        ("img", np.uint8, (512, 512)),
        ("ghdr", np.uint8, HEADER_MAX_SIZE),
        ("gt", np.uint64),
        ("geom", np.float32, (470, 3)),
        ("calib", np.float32, (3, 3)),
        ("score", np.int64),
    ]
)

# venus (cropped and resized)
frameCrop_dtype = np.dtype(
    [
        ("hdr", np.uint8, HEADER_MAX_SIZE),
        ("t", np.uint64),
        ("ser", np.uint8, HEADER_MAX_SIZE),
        ("n", np.uint64),
        ("img", np.uint8, (450, 550)),
        ("ghdr", np.uint8, HEADER_MAX_SIZE),
        ("gt", np.uint64),
        ("geom", np.float32, (470, 3)),
        ("calib", np.float32, (3, 3)),
        ("score", np.int64),
    ]
)


