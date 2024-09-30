from __future__ import annotations

from shiny import App, ui, render, reactive
import shinyswatch

COL_GRP1 = "#0a9396"
COL_GRP2 = "#f95738"
COL_LDA_LINE = "#ee9b00"
COL_L1 = "#6c757d"
COL_DIM = "#e9d8a6"
COL_DIST = "#06d6a0"
ALPHA_DIM = 0.4
LINE_DIM = 0.5

app_ui = ui.page_fluid(
    shinyswatch.theme.cosmo(),
    ui.br(),
    ui.HTML(
        "<div style='background-color: skyblue; padding: 10px; font-size:2em;'>"
        "Predicting Philadelphia Chromosome-Positive Acute Lymphoblastic Leukemia"
        "</div>"
    ),
    ui.br(),
    ui.p(
        "Welcome to our online calculator for predicting Philadelphia Chromosome-Positive Acute Lymphoblastic Leukemia (Ph+ ALL).",
        " This tool utilizes 17 critical blood parameters and a demographic feature, including MXD#, age, NEUT#, and others, to help users assess their potential risk for the disease.",
        " We invite you to explore the features and gain insights into your health!",
    ),
    # ui.p(
    #     "Let's see this in action.",
    #     "Here are two groups of points. ",
    #     "The objective is to find a line which best separates the two groups. ",
    # ),
    ui.row(
        ui.column(
            2,
            ui.br(),
            ui.panel_well(
                # 1
                ui.input_numeric(
                    id="MXD_count",
                    label="MXD# (10^9/L)",
                    value=0.0,
                    min=0.0,
                    max=100000.0
                ),
                # 2
                ui.input_slider(
                    "Age", "Age (years)", 0, 150, 75, step=1
                ),
                # 3
                ui.input_numeric(
                    id="NEUT_count",
                    label="NEUT# (10^9/L)",
                    value=0.0,
                    min=0.0,
                    max=100000.0
                ),
                # 4
                ui.input_numeric(
                    id="CREA",
                    label="CREA (µmol/L)",
                    value=0.0,
                    min=0.0,
                    max=100000.0
                ),
                # 5
                ui.input_numeric(
                    id="GGT",
                    label="GGT (U/L)",
                    value=0.0,
                    min=0.0,
                    max=100000.0
                ),
                # 6
                ui.input_numeric(
                    id="LYM_count",
                    label="LYM# (10^9/L)",
                    value=0.0,
                    min=0.0,
                    max=100000.0
                ),
            ),
            ui.br(),
            ui.row(
                ui.column(
                    2,
                    ui.input_action_button("predict", "Predict")
                ),
            ),
        ),
        ui.column(
            2,
            ui.br(),
            ui.panel_well(
                # 7
                ui.input_numeric(
                    id="ALT",
                    label="ALT (U/L)",
                    value=0.0,
                    min=0.0,
                    max=100000.0
                ),
                # 8
                ui.input_numeric(
                    id="MXD_per",
                    label="MXD%",
                    value=0.0,
                    min=0.0,
                    max=100.0
                ),
                # 9
                ui.input_numeric(
                    id="PLT",
                    label="PLT (10^9/L)",
                    value=0.0,
                    min=0.0,
                    max=10000.0
                ),
                # 10
                ui.input_numeric(
                    id="PT",
                    label="PT (seconds)",
                    value=0.0,
                    min=0.0,
                    max=10000.0
                ),
                # 11
                ui.input_numeric(
                    id="NA",
                    label="NA (mmol/L)",
                    value=0.0,
                    min=0.0,
                    max=10000.0
                ),
                # 12
                ui.input_numeric(
                    id="BASO_count",
                    label="BASO# (10^9/L)",
                    value=0.0,
                    min=0.0,
                    max=10000.0
                ),
            ),
        ),
        ui.column(
            2,
            ui.br(),
            ui.panel_well(
                # 13
                ui.input_numeric(
                    id="WBC",
                    label="WBC (10^9/L)",
                    value=0.0,
                    min=0.0,
                    max=10000.0
                ),
                # 14
                ui.input_numeric(
                    id="TP",
                    label="TP (g/L)",
                    value=0.0,
                    min=0.0,
                    max=10000.0
                ),
                # 15
                ui.input_numeric(
                    id="AST",
                    label="AST (U/L)",
                    value=0.0,
                    min=0.0,
                    max=10000.0
                ),
                # 16
                ui.input_numeric(
                    id="LYM_per",
                    label="LYM%",
                    value=0.0,
                    min=0.0,
                    max=100.0
                ),
                # 17
                ui.input_numeric(
                    id="TT",
                    label="TT (seconds)",
                    value=0.0,
                    min=0.0,
                    max=10000.0
                ),
                # 18
                ui.input_numeric(
                    id="MCH",
                    label="MCH (pg)",
                    value=0.0,
                    min=0.0,
                    max=10000.0
                ),
                # ui.input_action_button("predict", "Predict")
                # ui.input_checkbox("show", "Show image?", value=True)
            ),
        ),
        # ui.br(),
        # ui.br(),
        # ui.row(
        #         ui.output_text("prediction")
        # ),
        ui.column(
            6,
            # ***为什么你不显示出来啊啊啊啊啊啊啊***
            # ui.img(src="images/output.jpg"),
            # ui.page_fluid(
            #     ui.input_file("file_input", "Random Forest Feature Selection Scores"),
            ui.output_image("image", height="550px",  width="600px"),
            # ui.output_image("image", height="480px"),
            # ),
            ui.HTML(
                f"<div style='text-align: justify;'>We performed feature selection from the original <span style='color:{COL_LDA_LINE};'>49</span> parameters using random forest and Gini importance measures."
            ),
            ui.HTML(
                f" The resulting bar chart illustrates <span style='color:{COL_LDA_LINE};'>33</span> selected parameters with importance greater than 0, while the final model was implemented with the top <span style='color:{COL_LDA_LINE};'>18</span> parameters, including MXD#, age, NEUT#, and others."
            ),
            ui.HTML(
                " This visual representation highlights the key factors that contribute significantly to the prediction model, aiding in better understanding of their roles in the diagnosis."
            ),
        ),
    ),
    # ui.br(),
    # ui.br()∂,
    # ui.row(
    #     ui.column(
    #         2,
    #         ui.input_action_button("predict", "Predict")
    #     ),
    # ),
    ui.row(
        ui.column(
            6,
            ui.panel_well(
                ui.output_text("prediction")
            ),
        ),
    ),
    ui.br(),
    ui.br(),
    ui.row(
        ui.HTML(
            "<div style='text-align: center; color: gray; font-size:0.9em;'> Created using Shiny for Python | The Second Affiliated Hospital of Army Medical University</a> | Sep '26</div>"
        )
    ),
)

from pathlib import Path
import pandas as pd
import joblib

# current_dir = os.path.dirname(os.path.abspath(__file__))
# model_path = os.path.join(current_dir, "04-dashboard-tips/models/lr_with_AL.pkl")
# scaler_path = os.path.join(current_dir, "04-dashboard-tips/models/lr_scaler_with_AL.pkl")

lr_with_AL = joblib.load("models/lr_with_AL.pkl")
lr_scaler = joblib.load("models/lr_scaler_with_AL.pkl")

here = Path("images/output.jpg").parent

# 定义服务器函数
# ***现在读不进数据***
def server(input, output, session):
    @output
    @render.text
    @reactive.event(input.predict, ignore_none=False)
    def prediction():

        data = {
            "MXD#": input.MXD_count(),
            "age": input.Age(),
            "NEUT#": input.NEUT_count(),
            "CREA": input.CREA(),
            "GGT": input.GGT(),
            "LYM#": input.LYM_count(),
            "ALT": input.ALT(),
            "MXD%": input.MXD_per(),
            "PLT": input.PLT(),
            "PT": input.PT(),
            "NA": input.NA(),
            "BASO#": input.BASO_count(),
            "WBC": input.WBC(),
            "TP": input.TP(),
            "AST": input.AST(),
            "LYM%": input.LYM_per(),
            "TT": input.TT(),
            "MCH": input.MCH()
        }

        data = pd.DataFrame([data])

        # 计算等于0的参数数量
        zero_count = (data == 0).sum(axis = 1)

        # 检查条件并返回None或数据
        if zero_count.iloc[0] >= 6:
            return "Invalid inputs. Please Check your inputs!"

        data = lr_scaler.transform(data)

        prediction = lr_with_AL.predict(data)

        if prediction == 0:
            text = "Ph+ ALL"
            prob = lr_with_AL.predict_proba(data)[:, 0]
        else:
            text = "Ph- ALL"
            prob = lr_with_AL.predict_proba(data)[:, 1]

        result = f"You are predicted to be a {text} patient with a probability of {prob[0]:.2f}."

        return result

    @output
    @render.image
    def image():
        img = {"src": here / "output.jpg", "width": "700px"}
        # img = {"src": here / "output.jpg", "width": "600px"}

        return img

app = App(app_ui, server)

if __name__ == "__main__":
    app.run()