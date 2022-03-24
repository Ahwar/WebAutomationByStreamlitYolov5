# decompyle3 version 3.8.0
# Python bytecode 3.7.0 (3394)
# Decompiled from: Python 3.7.0 (v3.7.0:1bf9cc5093, Jun 27 2018, 04:59:51) [MSC v.1914 64 bit (AMD64)]
# Embedded file name: main_with_detection.py
import os
from datetime import datetime, timedelta
import time, cv2, sys, socket, numpy as np, shutil, time, onnxruntime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as EC

model_name = "models/yolov3_captcha_20210624"
"""
if datetime.now().strftime('%Y-%m') == '2021-06' or datetime.now().strftime('%Y-%m') == '2021-07':
    session = onnxruntime.InferenceSession(f"{model_name}.onnx")
    inputName = session.get_inputs()[0].name
else:
    print('Competition period has ended!')
    time.sleep(30)
    sys.exit()
"""


session = onnxruntime.InferenceSession(f"{model_name}.onnx")
inputName = session.get_inputs()[0].name
print(f"Detection model {model_name} loaded")
NUM_CLASS = {
    0: "E",
    1: "M",
    2: "C",
    3: "K",
    4: "X",
    5: "U",
    6: "W",
    7: "F",
    8: "B",
    9: "H",
    10: "6",
    11: "T",
    12: "4",
    13: "7",
    14: "Q",
    15: "J",
    16: "P",
    17: "G",
    18: "8",
    19: "S",
    20: "A",
    21: "5",
    22: "Y",
    23: "2",
    24: "V",
    25: "D",
    26: "R",
    27: "L",
    28: "3",
    29: "N",
    30: "9",
}


def WaitToLoadWD(driver, time, locator):
    """wait for element by locator parameter to get loaded"""
    WebDriverWait(driver, time).until(
        lambda driver: driver.find_element(By.XPATH, value=locator)
    )
    element = driver.find_element(By.XPATH, value=locator)

    return element


def WaitToBeClickable(driver, time, locator):
    """wait for element by locator parameter to be clickable"""
    element = WebDriverWait(driver, time).until(
        EC.element_to_be_clickable((By.XPATH, locator))
    )
    return element


def Move_to_web_element(driver, element):
    actions = ActionChains(driver)
    actions.move_to_element_with_offset(element, 5, 5).perform()


def SimpleClick(driver, el):
    """Simply click on the element"""
    el.click()


def CheckExistsByXpath(driver, locator):
    WebDriverWait(driver, 10).until(
        lambda driver: driver.find_element(By.XPATH, value=locator)
    )
    element = driver.find_element(By.XPATH, value=locator)
    return element


def SimpleSelect(driver, element, text):
    """Select drop down menu by visible text"""
    select = Select(element)
    select.select_by_visible_text(text)


def image_preprocess(image, target_size, gt_boxes=None):
    ih, iw = target_size
    h, w, _ = image.shape
    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))
    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_paded[dh : nh + dh, dw : nw + dw, :] = image_resized
    image_paded = image_paded / 255.0
    if gt_boxes is None:
        return image_paded
    gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
    gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
    return (image_paded, gt_boxes)


def postprocess_boxes(pred_bbox, original_image, input_size, score_threshold):
    valid_scale = [0, np.inf]
    pred_bbox = np.array(pred_bbox)
    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]
    pred_coor = np.concatenate(
        [
            pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
            pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5,
        ],
        axis=(-1),
    )
    org_h, org_w = original_image.shape[:2]
    resize_ratio = min(input_size / org_w, input_size / org_h)
    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2
    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio
    pred_coor = np.concatenate(
        [
            np.maximum(pred_coor[:, :2], [0, 0]),
            np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1]),
        ],
        axis=(-1),
    )
    invalid_mask = np.logical_or(
        pred_coor[:, 0] > pred_coor[:, 2], pred_coor[:, 1] > pred_coor[:, 3]
    )
    pred_coor[invalid_mask] = 0
    bboxes_scale = np.sqrt(
        np.multiply.reduce((pred_coor[:, 2:4] - pred_coor[:, 0:2]), axis=(-1))
    )
    scale_mask = np.logical_and(
        valid_scale[0] < bboxes_scale, bboxes_scale < valid_scale[1]
    )
    classes = np.argmax(pred_prob, axis=(-1))
    scores = pred_conf * pred_prob[(np.arange(len(pred_coor)), classes)]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]
    return np.concatenate(
        [coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=(-1)
    )


def bboxes_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
    boxes1_area = (boxes1[(Ellipsis, 2)] - boxes1[(Ellipsis, 0)]) * (
        boxes1[(Ellipsis, 3)] - boxes1[(Ellipsis, 1)]
    )
    boxes2_area = (boxes2[(Ellipsis, 2)] - boxes2[(Ellipsis, 0)]) * (
        boxes2[(Ellipsis, 3)] - boxes2[(Ellipsis, 1)]
    )
    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[(Ellipsis, 0)] * inter_section[(Ellipsis, 1)]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)
    return ious


def nms(bboxes, iou_threshold, sigma=0.3, method="nms"):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []
    for cls in classes_in_img:
        cls_mask = bboxes[:, 5] == cls
        cls_bboxes = bboxes[cls_mask]
        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate(
                [cls_bboxes[:max_ind], cls_bboxes[max_ind + 1 :]]
            )
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=(np.float32))
            if not method in ("nms", "soft-nms"):
                raise AssertionError
            else:
                if method == "nms":
                    iou_mask = iou > iou_threshold
                    weight[iou_mask] = 0.0
                if method == "soft-nms":
                    weight = np.exp(-(1.0 * iou**2 / sigma))
                cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
                score_mask = cls_bboxes[:, 4] > 0.0
                cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes


end_byte = b"\x99\x99\x99\x98"
height = 90
width = 300
CHANNELS = 3
CAPTCHA_LEN = 6
characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CLASSES = len(characters)


def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (width, height))
    _, img_encoded = cv2.imencode(".png", img)
    data = img_encoded.tostring()
    return data


def main(data):
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = "78.56.201.100"
    port = 22000
    try:
        soc.connect((host, port))
    except:
        print("Connection error")
        return "noconn"
    else:
        sending_data = data + end_byte
        soc.sendall(sending_data)
        answer = soc.recv(10).decode()
        return answer


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def decode(y):
    y = np.argmax((np.array(y)), axis=2)[:, 0]
    return "".join([characters[x] for x in y])


def DoTheDetection(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_data = image_preprocess(np.copy(image), [416, 416])
    image_data = image_data[(np.newaxis, ...)].astype(np.float32)
    pred_bbox = session.run(None, {inputName: image_data})
    pred_bbox = [np.reshape(x, (-1, x.shape[(-1)])) for x in pred_bbox]
    pred_bbox = np.concatenate(pred_bbox, axis=0)
    bboxes = postprocess_boxes(pred_bbox, image, input_size=416, score_threshold=0.3)
    bboxes = nms(bboxes, iou_threshold=0.45, method="nms")
    boxes = []
    for bbox in bboxes:
        boxes.append(
            [
                bbox[0].astype(int),
                bbox[1].astype(int),
                bbox[2].astype(int),
                bbox[3].astype(int),
                bbox[4],
                NUM_CLASS[int(bbox[5])],
                bbox[0].astype(int) + (bbox[2].astype(int) - bbox[0].astype(int)) / 2,
            ]
        )

    sorted_list = sorted(boxes, key=(lambda x: x[(-1)]))
    captcha_name = ""
    for i in sorted_list:
        captcha_name += str(i[5])

    return captcha_name


def get_captcha(driver, element, path, INPUT):
    location = element.location_once_scrolled_into_view
    element.screenshot(f"body{INPUT}.png")
    image = cv2.imread(f"body{INPUT}.png", 1)
    resized = cv2.resize(image, (300, 90))
    t1 = time.time()
    result = DoTheDetection(image)
    print("Detection time", time.time() - t1)
    return result


driver_path = "C:/Users/zubaria.farrukh/.wdm/drivers/chromedriver/win32/99.0.4844.51/chromedriver.exe"
download_path = os.getcwd()
profile_path = ""
eGOV = "https://www.e-gov.az/az/services/read/3766/0"
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.get(eGOV)
driver.set_window_size(1080, 1080)


# element = driver.find_element(By.XPATH, value="//*[@class='col-md-4']//select[@class='form-control']")
# element.click()
# //option[@value='719a8862-4827-4485-829c-fc4f81b7d100']
# element = driver.find_element(By.XPATH, value="//option[@value='719a8862-4827-4485-829c-fc4f81b7d100']")


if not os.path.exists("SOLVED_CAPTCHAS"):
    os.makedirs("SOLVED_CAPTCHAS")
if not os.path.exists("BAD_CAPTCHA"):
    os.makedirs("BAD_CAPTCHA")


INPUT = input("Write case number from 1 to 10: ")
if INPUT == "1":
    case, case2 = (1, 1)
if INPUT == "2":
    case, case2 = (1, 2)
if INPUT == "3":
    case, case2 = (1, 3)
if INPUT == "4":
    case, case2 = (2, 1)
if INPUT == "5":
    case, case2 = (2, 2)
if INPUT == "6":
    case, case2 = (2, 3)
if INPUT == "7":
    case, case2 = (1, 3)
if INPUT == "8":
    case, case2 = (2, 3)
if INPUT == "9":
    Bina = input("Bina nömrəsi: ")
    Giris = input("Giriş nömrəsi: ")
    Mertebe = input("Mərtəbə nömrəsi: ")
    Menzil = input("Mənzil nömrəsi: ")
    case = 1
if INPUT == "10":
    # Bina = input("Bina nömrəsi: ")
    # Giris = input("Giriş nömrəsi: ")
    # Mertebe = input("Mərtəbə nömrəsi: ")
    # Menzil = input("Mənzil nömrəsi: ")
    case = 2
input("Press Enter to solve CAPTCHA...")


# ------------------------------------------------------------- drop down choose
driver.switch_to.window(driver.window_handles[(-1)])
driver.switch_to.default_content()
iframe = WaitToLoadWD(driver, 30, "//iframe[@id='ServiceScreen']")
driver.switch_to.frame(0)

"""#WebDriverWait(driver, 30).until(lambda driver: driver.find_element(By.XPATH, value="//*[@id='options']/form/div[3]/div[2]/div[1]/div[4]/div/div[2]/select"))
el_select = WaitToLoadWD(driver,20,"//*[@id='options']/form/div[3]/div[2]/div[1]/div[4]/div/div[2]/select")
select = Select(el_select)
#scroll to element
Move_to_web_element(driver,el_select)
# select by visible text
select.select_by_visible_text('Yasamal Yaşayış Kompleksinin ikinci mərhələsi')"""
# -------------------------------------------------------------------------------


def first_loop():
    if case == 1 or INPUT == "9":
        Odenis = WaitToLoadWD(
            driver, 5, "//span[contains(text(), 'Öz vəsaiti hesabına')]"
        )
    else:
        Odenis = WaitToLoadWD(
            driver, 5, "//span[contains(text(), 'İpoteka krediti hesabına')]"
        )
    if Odenis:
        if Odenis.get_attribute("class") != "choice choice-selected":
            SimpleClick(driver, Odenis)
            SimpleClick(driver, Odenis)
    if INPUT == "9" or INPUT == "10":
        Menzil_2 = WaitToLoadWD(driver, 5, "//span[contains(text(), 'Ünvan üzrə')]")
    else:
        Menzil_2 = WaitToLoadWD(
            driver, 5, "//span[contains(text(), 'Parametrlər üzrə')]"
        )
    if Menzil_2:
        if Menzil_2.get_attribute("class") != "choice choice-selected":
            SimpleClick(driver, Menzil_2)
            SimpleClick(driver, Menzil_2)
    Layihe_3 = WaitToLoadWD(
        driver,
        5,
        "//option[contains(text(), 'Yasamal Yaşayış Kompleksinin ikinci mərhələsi')]",
    )
    if Layihe_3:
        if Layihe_3.get_attribute("class") != "choice choice-selected":
            SimpleClick(driver, Layihe_3)
            SimpleClick(driver, Layihe_3)
    CaptchaIMG = WaitToLoadWD(driver, 5, "//div[@class='col-md-4']/p[2]/img")
    answer = get_captcha(driver, CaptchaIMG, f"captcha{INPUT}.png", INPUT)
    CaptchaInput = CheckExistsByXpath(driver, "//input[@name='Captcha']")
    CaptchaInput.location_once_scrolled_into_view
    CaptchaInput.clear()
    CaptchaInput.send_keys(answer)
    print(answer)
    Next = CheckExistsByXpath(driver, "//button[@id='next']")
    if Next:
        SimpleClick(driver, Next)
    if len(answer) != 6:
        print("bad answer length")
        return True
    if INPUT == "9" or INPUT == "10":
        CaptchaIMG_ = WaitToLoadWD(
            driver, 2, "//div[@class='col-md-3']/div[6]/div/div/p[*]/img"
        )
    else:
        CaptchaIMG_ = WaitToLoadWD(
            driver, 2, "//div[@class='col-md-3']/div[5]/div/p[*]/img"
        )
    if CaptchaIMG_:
        print("finished first part")
        return False
    Captcha_ok = WaitToLoadWD(driver, 1, "//span[@data-valmsg-for='Captcha']")
    if Captcha_ok == False:
        return False
    return True


while True:
    result = first_loop()
    if not result:
        break

# -----------------------------------------------------second screen with parameters

Bina = 4
Giris = 27
Mertebe = 4
Menzil = 1
case = 2

driver.switch_to.window(driver.window_handles[(-1)])
driver.switch_to.default_content()
iframe = WaitToLoadWD(driver, 30, "//iframe[@id='ServiceScreen']")
driver.switch_to.frame(0)

# wait for select 1
WebDriverWait(driver, 10).until(
    EC.presence_of_element_located(
        (
            By.XPATH,
            "//*[@id='address']/form/div[3]/div[2]/div[1]/div[1]/div[1]/div/div[2]/select",
        )
    )
)
while 1:
    if INPUT == "9" or INPUT == "10":

        Option1 = WaitToBeClickable(
            driver,
            10,
            "//*[@id='address']/form/div[3]/div[2]/div[1]/div[1]/div[1]/div/div[2]/select",
        )
        if Option1:
            SimpleSelect(driver, Option1, str(Bina))
        Option2 = WaitToBeClickable(
            driver,
            10,
            "//*[@id='address']/form/div[3]/div[2]/div[1]/div[1]/div[2]/div/div[2]/select",
        )
        if Option2:
            SimpleSelect(driver, Option2, str(Giris))
        Option3 = WaitToBeClickable(
            driver,
            10,
            "//*[@id='address']/form/div[3]/div[2]/div[1]/div[1]/div[3]/div/div[2]/select",
        )
        if Option3:
            SimpleSelect(driver, Option3, str(Mertebe))
        Option4 = WaitToBeClickable(
            driver,
            10,
            "//*[@id='address']/form/div[3]/div[2]/div[1]/div[1]/div[4]/div/div[2]/select",
        )
        if Option4:
            SimpleSelect(driver, Option4, str(Menzil))
    else:
        if INPUT == "1" or "2" or "4" or "5" or "7":
            mertebeli7 = WaitToLoadWD(
                driver, 5, "//div[@class='col-md-3']/div[1]/ul/li[1]"
            )
            if mertebeli7:
                mertebeli7.location_once_scrolled_into_view
                if mertebeli7.get_attribute("class") != "list-group-item active":
                    SimpleClick(driver, mertebeli7)
            if INPUT == "1" or (
                INPUT == "2"
                or INPUT == "4"
                or INPUT == "5"
                or INPUT == "7"
                or INPUT == "8"
            ):
                if INPUT == "1" or INPUT == "2" or INPUT == "4" or INPUT == "5":
                    a, b = (2, 6)
                else:
                    if INPUT == "7" or (INPUT == "8"):
                        a, b = (7, 7)
                Option1 = WaitToLoadWD(
                    driver,
                    5,
                    f"//div[@class='col-md-3']/div[2]/div/div/select[1]/option[{a}]",
                )
                if Option1:
                    SimpleClick(driver, Option1)
                Option2 = WaitToLoadWD(
                    driver,
                    5,
                    f"//div[@class='col-md-3']/div[2]/div/div/select[2]/option[{b}]",
                )
                if Option2:
                    SimpleClick(driver, Option2)
            otaqlı = WaitToLoadWD(
                driver, 5, f"//div[@class='col-md-3']/div[3]/ul/li[{case2}]"
            )
            if otaqlı:
                otaqlı.location_once_scrolled_into_view
                if otaqlı.get_attribute("class") != "list-group-item active":
                    SimpleClick(driver, otaqlı)
    if INPUT == "9" or INPUT == "10":
        CaptchaIMG = WaitToLoadWD(
            driver, 5, "//div[@class='col-md-3']/div[6]/div/div/p[*]/img"
        )
    else:
        CaptchaIMG = WaitToLoadWD(
            driver, 5, "//div[@class='col-md-3']/div[5]/div/p[*]/img"
        )
    if CaptchaIMG:
        CaptchaIMG.location_once_scrolled_into_view
        size = CaptchaIMG.size
        while size["width"] == 0:
            CaptchaIMG = WaitToLoadWD(
                driver, 5, "//div[@class='col-md-3']/div[5]/div/p[*]/img"
            )
            CaptchaIMG.location_once_scrolled_into_view
            size = CaptchaIMG.size

        answer = get_captcha(driver, CaptchaIMG, f"captcha{INPUT}.png", INPUT)
        print(answer)
        if INPUT == "9" or INPUT == "10":
            CaptchaInput = CheckExistsByXpath(
                driver, "//div[@class='col-md-3']/div[6]/div/div/p[1]/input"
            )
        else:
            CaptchaInput = CheckExistsByXpath(
                driver, "//div[@class='col-md-3']/div[5]/div/p[1]/input"
            )
        CaptchaInput.location_once_scrolled_into_view
        CaptchaInput.clear()
        CaptchaInput.send_keys(answer)
        if INPUT == "9" or INPUT == "10":
            Next = CheckExistsByXpath(
                driver, "//div[@class='col-md-3']/div[7]/div/button"
            )
        else:
            Next = CheckExistsByXpath(
                driver, "//div[@class='col-md-3']/div[6]/div/button"
            )
        if Next:
            SimpleClick(driver, Next)
        if len(answer) != 6:
            print("bad answer length")
            continue

    if INPUT == "9" or (INPUT == "10"):
        Next = driver.find_element(By.CSS_SELECTOR, value="a#next")
    if Next.is_enabled():
        try:
            SimpleClick(driver, Next)
            print(time.time())
        finally:
            driver.implicitly_wait(0.3)
            print(time.time())
            break

print("finished second part")
