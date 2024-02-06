import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pipeline import process_and_predict

# 预先定义 color_map 以便格式化百分比标签
color_map = {0: "Blue", 1: "Red", 2: "Yellow", 3: "Green", 'others': "Not Sketched"}

def upload_action():
    filename = filedialog.askopenfilename()
    if filename:
        fig, pred_percentage = process_and_predict(filename)
        update_result(fig, pred_percentage)

def update_result(fig, pred_percentage):
    global percentage_label
    global canvas

    # 格式化百分比标签，对于不存在于 color_map 中的键使用 'Unknown' 标签
    percentage_text = "\n".join(
        f"{color_map.get(i, 'Not Sketched')} percentage: {p:.2f}%"
        for i, p in enumerate(pred_percentage)
    )

    # 如果已经有百分比标签，更新它；否则创建一个新的
    if 'percentage_label' in globals():
        percentage_label.config(text=percentage_text)
    else:
        percentage_label = tk.Label(root, text=percentage_text, justify=tk.LEFT, anchor="w")
        percentage_label.pack()

    # 如果已经有画布，更新它；否则创建一个新的
    if 'canvas' in globals():
        canvas.figure = fig
        canvas.draw()
    else:
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack()


# 创建 Tkinter 窗口和控件...
root = tk.Tk()
root.title("图像处理应用")

upload_button = tk.Button(root, text="上传图片", command=upload_action)
upload_button.pack()

# 应用程序的主循环
root.mainloop()
