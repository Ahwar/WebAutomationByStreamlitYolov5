from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pylightxl as xl
import time
from webdriver_manager.chrome import ChromeDriverManager

try:
    db = xl.readxl(fn="input.xlsx")

    values = db.ws(ws="Sheet1").row(row=2)
    values

    odenis = str(values[0]).strip()
    menzil = str(values[1]).strip()
    layiha = str(values[2]).strip()
    bina = str(values[3]).strip()
    giris = str(values[4]).strip()
    martaba = str(values[5]).strip()
    manzil = str(values[6]).strip()
    mar_1 = str(values[7]).strip()
    mar_2 = str(values[8]).strip()
except:
    input("Input file not Found or not in Correct Format ...")
options = webdriver.ChromeOptions()
options.add_argument("start-maximized")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option("useAutomationExtension", False)
driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)

url = "https://www.e-gov.az/az/login/index/"
driver.get(url)
# https://www.e-gov.az/az/services/read/3766/0
input("\n\nOpen Captcha Page and Press Enter ...")

try:
    iframe = driver.find_element_by_xpath("//iframe")
    driver.switch_to.frame(iframe)
except:
    pass
while True:
    try:
        driver.find_elements_by_xpath("//div[@class='thumbnail']")[2].click()
    except:
        pass
    try:
        element = WebDriverWait(driver, 1).until(
            EC.element_to_be_clickable(
                (By.XPATH, "//*[contains(text(),'{}')]".format(odenis))
            )
        )
        break
    except:
        pass
driver.find_element_by_xpath("//*[contains(text(),'{}')]".format(odenis)).click()
driver.find_element_by_xpath("//*[contains(text(),'{}')]".format(menzil)).click()
driver.find_element_by_xpath("//*[contains(text(),'{}')]".format(layiha)).click()

element = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, "next")))
driver.find_element_by_id("next").click()

if "Parametr" in menzil:
    element = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//*[contains(text(),'7 mərtəbəli')]"))
    )
else:
    element = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//option[text()='1']"))
    )
try:
    driver.find_element_by_xpath("//*[contains(text(),'Bina nömrəsi')]")
    try:
        bina_nom = driver.find_element_by_xpath(
            "//*[contains(text(),'Bina nömrəsi')]/../..//select/option[@value='{}']".format(
                bina
            )
        ).click()
    except:
        pass
    try:
        giris_nom = driver.find_element_by_xpath(
            "//*[contains(text(),'Giriş nömrəsi')]/../..//select/option[@value='{}']".format(
                giris
            )
        ).click()
    except:
        pass
    try:
        mar_nom = driver.find_element_by_xpath(
            "//*[contains(text(),'Mərtəbə nömrəsi')]/../..//select/option[@value='{}']".format(
                martaba
            )
        ).click()
    except:
        pass
    try:
        manz_nom = driver.find_element_by_xpath(
            "//*[contains(text(),'Mənzil nömrəsi')]/../..//select/option[@value='{}']".format(
                manzil
            )
        ).click()
    except:
        pass
    try:
        element = WebDriverWait(driver, 3).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(),'Axtar')]"))
        )
    except:
        pass
    try:
        next_bt = driver.find_element_by_xpath(
            "//button[contains(text(),'Axtar')]"
        ).click()
    except:
        pass
    try:
        element = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "next"))
        )
        for i in range(3):
            try:
                driver.find_element_by_id("next").click()
                break
            except:
                pass
            time.sleep(1)
    except:
        pass
except:
    try:
        driver.find_element_by_xpath("//*[contains(text(),'7 mərtəbəli')]").click()
    except:
        pass
    try:
        driver.find_element_by_xpath("//*[contains(text(),'3 otaqlı')]").click()
    except:
        pass
    try:
        driver.find_element_by_xpath(
            "//select[contains(@class, 'min')]//option[text()='{}']".format(mar_1)
        ).click()
    except:
        pass
    try:
        driver.find_element_by_xpath(
            "//select[contains(@class, 'max')]//option[text()='{}']".format(mar_2)
        ).click()
    except:
        pass
    try:
        driver.find_element_by_xpath("//li[text()='1']").click()
    except:
        pass
    try:
        driver.find_element_by_xpath("//li[text()='2']").click()
    except:
        pass
    try:
        element = WebDriverWait(driver, 3).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(),'Axtar')]"))
        )
    except:
        pass
    try:
        next_bt = driver.find_element_by_xpath(
            "//button[contains(text(),'Axtar')]"
        ).click()
    except:
        pass
