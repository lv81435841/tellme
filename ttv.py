"""
 作者 lgf
 日期 2023/5/12
"""
from msedge.selenium_tools import Edge, EdgeOptions
import re
import time

# 创建一个 Microsoft Edge 浏览器驱动程序实例
options = EdgeOptions()
options.use_chromium = True
driver = Edge(executable_path='edgedriver_win64/msedgedriver', options=options)
# 打开网页
driver.get('https://models.aminer.cn/sign/')

# 等待页面加载完成
driver.implicitly_wait(10)

# 查找包含文本框的父元素
parent_element = driver.find_element_by_xpath('//div[@class="ant-form-item-control-input-content"]')

# 在父元素中查找文本框并输入文本
text_input = parent_element.find_element_by_xpath('.//textarea')
print(text_input.text)
text_input.send_keys('Selenium')
print(text_input.text)

# 查找提交按钮
submit_button = driver.find_element_by_xpath('/html/body/div[1]/div/section/div/main/div/div[2]/div/div[1]/div[2]/div/div/form/div[2]/div/div/div/div/div/button')

submit_button.click()

# 等待页面加载完成
time.sleep(5)

# 获取页面内容
page_source = driver.page_source

# 这里你可以使用正则表达式或其他方法来筛选出所有的.mp4链接
# 并将它们存储到一个列表中
# 使用正则表达式匹配所有的.mp4链接
mp4_links = re.findall(r'href=[\'"]?([^\'" >]+\.mp4)', page_source)

# 输出所有匹配到的链接
for link in mp4_links:
    print(link)

# 关闭浏览器
driver.quit()