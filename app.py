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
        " This tool utilizes 10 critical blood parameters and a demographic feature, including age, NEUT#, AST, and others, to help users assess their potential risk for the disease.",
        " We invite you to explore the features and gain insights into your health!",
    ),
    # ui.p(
    #     "Let's see this in action.",
    #     "Here are two groups of points. ",
    #     "The objective is to find a line which best separates the two groups. ",
    # ),
    ui.row(
        ui.column(
            6,
            ui.br(),
            ui.panel_well(
                # 1
                ui.input_slider(
                    "Age", "Age (years)", 0, 150, 75, step=1
                ),
                # 2
                ui.input_numeric(
                    id="NEUT_count",
                    label="NEUT# (10^9/L)",
                    value=0.0,
                    min=0.0,
                    max=100000.0
                ),
                # 3
                ui.input_numeric(
                    id="AST",
                    label="AST (U/L)",
                    value=0.0,
                    min=0.0,
                    max=10000.0
                ),
                # 4
                ui.input_numeric(
                    id="MXD_count",
                    label="MXD# (10^9/L)",
                    value=0.0,
                    min=0.0,
                    max=100000.0
                ),
                # 5
                ui.input_numeric(
                    id="MXD_per",
                    label="MXD%",
                    value=0.0,
                    min=0.0,
                    max=100.0
                ),
            ),
            ui.br(),
            ui.row(
                ui.column(
                    3,
                    ui.input_action_button("predict", "Predict")
                ),
            ),
        ),
        ui.column(
            6,
            ui.br(),
            ui.panel_well(
                # 6
                ui.input_numeric(
                    id="GGT",
                    label="GGT (U/L)",
                    value=0.0,
                    min=0.0,
                    max=100000.0
                ),
                # 7: unit
                ui.input_numeric(
                    id="UA",
                    label="UA (umol/L)",
                    value=0.0,
                    min=0.0,
                    max=100000.0
                ),
                # 8: unit
                ui.input_numeric(
                    id="UREA",
                    label="UREA (umol/L)",
                    value=0.0,
                    min=0.0,
                    max=10000.0
                ),
                # 9
                ui.input_numeric(
                    id="ALT",
                    label="ALT (U/L)",
                    value=0.0,
                    min=0.0,
                    max=100000.0
                ),
                # 10
                ui.input_numeric(
                    id="TT",
                    label="TT (seconds)",
                    value=0.0,
                    min=0.0,
                    max=10000.0
                ),
            ),
        ),
        # ui.br(),
        # ui.br(),
        # ui.row(
        #         ui.output_text("prediction")
        # ),
        # ui.column(
        #     6,
        #     # ***为什么你不显示出来啊啊啊啊啊啊啊***
        #     # ui.img(src="images/output.jpg"),
        #     # ui.page_fluid(
        #     #     ui.input_file("file_input", "Random Forest Feature Selection Scores"),
        #     ui.output_image("image"),
        #     # ui.output_image("image", height="480px"),
        #     # ),
        #     ui.HTML(
        #         f"<div style='text-align: justify;'>We performed feature selection from the original <span style='color:{COL_LDA_LINE};'>49</span> parameters using random forest and Gini importance measures."
        #     ),
        #     ui.HTML(
        #         f" The resulting bar chart illustrates <span style='color:{COL_LDA_LINE};'>33</span> selected parameters with importance greater than 0, while the final model was implemented with the top <span style='color:{COL_LDA_LINE};'>18</span> parameters, including MXD#, age, NEUT#, and others."
        #     ),
        #     ui.HTML(
        #         " This visual representation highlights the key factors that contribute significantly to the prediction model, aiding in better understanding of their roles in the diagnosis."
        #     ),
        # ),
    ),
    # ui.br(),
    # ui.br(),
    # ui.row(
    #     ui.column(
    #         2,
    #         ui.input_action_button("predict", "Predict")
    #     ),
    # ),
    ui.br(),
    ui.row(
        ui.column(
            12,
            ui.panel_well(
                ui.output_text("prediction")
            ),
        ),
    ),
    ui.br(),
    ui.row(
        ui.column(
            12,
            # ui.output_image("image"),
            ui.HTML(
                f"<div style='text-align: justify;'>The figure presented below illustrates a <span style='color:{COL_LDA_LINE};'>SHAP force plot</span> generated from your sample data."
            ),
            ui.HTML(
                " This visualization demonstrates the granular contribution of individual features in driving the model's output from the base value to the final prediction (f(x))."
            ),
            ui.HTML(
                f" The red segments indicate features that push the prediction higher (i.e., <span style='color:{COL_LDA_LINE};'>Ph- ALL</span>), while blue segments represent features pushing the prediction lower (i.e., <span style='color:{COL_LDA_LINE};'>Ph+ ALL</span>)."
            ),
            ui.HTML(
                " The width of each segment corresponds to the magnitude of the feature's SHAP value."
            ),
            ui.output_plot("shap_plot"),
            # ui.HTML(
            #     f"<div style='text-align: justify;'>Please Note: The number dispalyed in the figure is being standarized.We performed feature selection from the original <span style='color:{COL_LDA_LINE};'>49</span> parameters using random forest and Gini importance measures."
            # ),
            # ui.HTML(
            #     f" The resulting bar chart illustrates <span style='color:{COL_LDA_LINE};'>33</span> selected parameters with importance greater than 0, while the final model was implemented with the top <span style='color:{COL_LDA_LINE};'>18</span> parameters, including MXD#, age, NEUT#, and others."
            # ),
            # ui.HTML(
            #     " This visual representation highlights the key factors that contribute significantly to the prediction model, aiding in better understanding of their roles in the diagnosis."
            # ),
        ),
    ),
    # ui.br(),
    # ui.br(),
    ui.row(
        ui.HTML(
            "<div style='text-align: center; color: gray; font-size:0.9em;'> Created using Shiny for Python | The Second Affiliated Hospital of Army Medical University</a> | Sep '26</div>"
        )
    ),
)

from pathlib import Path
import pandas as pd
import shap
import joblib
import numpy as np

# current_dir = os.path.dirname(os.path.abspath(__file__))
# model_path = os.path.join(current_dir, "04-dashboard-tips/models/lr_with_AL.pkl")
# scaler_path = os.path.join(current_dir, "04-dashboard-tips/models/lr_scaler_with_AL.pkl")

model_with_AL = joblib.load("models/CatBoost_AL_model.pkl")
scaler = joblib.load("models/CatBoost_AL_scaler.pkl")
# shap_explainer = joblib.load("models/shap_explainer.pkl")
# shap_values = joblib.load("models/shap_values.pkl")

# with open("models/AL_X_train.pkl", "rb") as f:
#     loaded_AL_X_train = pickle.load(f)

# loaded_AL_X_train = joblib.load("models/AL_X_train.joblib")

loaded_AL_X_train = np.load("models/AL_X_train.npy").tolist()

# here = Path("images/SHAP_plot.png").parent
here = Path("images/cropped_image.jpg").parent

# 定义服务器函数
def server(input, output, session):
    @output
    @render.text
    @reactive.event(input.predict, ignore_none=False)
    def prediction():

        if not input.predict():
            return 'Please enter numerical values and press the "Predict" button!'

        data = {
            "ALT": input.ALT(),
            "AST": input.AST(),
            "GGT": input.GGT(),
            "MXD#": input.MXD_count(),
            "MXD%": input.MXD_per(),
            "NEUT#": input.NEUT_count(),
            "TT": input.TT(),
            "UA": input.UA(),
            "UREA": input.UREA(),
            "age": input.Age(),
        }

        data = pd.DataFrame([data])
        # print("Original Data:", data)

        # filter_list = [data["NEUT#"], data["MXD#"], data["MXD%"]]
        # filter_zero_count = filter_list.count(0)

        others_df = data.drop(["NEUT#", "MXD#", "MXD%"], axis = 1)
        others_zero_count = others_df[others_df == 0].count().sum()

        if others_zero_count > 0:
            return "Invalid inputs. Please check your inputs!"

        # zero_count = data[data == 0].count().sum()
        # # print("Zero Count:", zero_count)
        #
        # if zero_count > 0:
        #     return "Invalid inputs. Please check your inputs!"

        # if zero_count >= 3:
        #     return None
        # if 0 < zero_count < 3:
        #     return "Invalid inputs. Please check your inputs!"

        data = scaler.transform(data)

        prediction = model_with_AL.predict(data)

        if prediction == 0:
            text = "Ph+ ALL"
            prob = model_with_AL.predict_proba(data)[:, 0]
        else:
            text = "Ph- ALL"
            prob = model_with_AL.predict_proba(data)[:, 1]

        result = f"You are predicted to be a {text} patient with a probability of {prob[0]:.2f}."

        return result

    @output
    @render.plot
    @reactive.event(input.predict, ignore_none=False)
    def shap_plot():

        if not input.predict():
            return None

        data = {
            "ALT": input.ALT(),
            "AST": input.AST(),
            "GGT": input.GGT(),
            "MXD#": input.MXD_count(),
            "MXD%": input.MXD_per(),
            "NEUT#": input.NEUT_count(),
            "TT": input.TT(),
            "UA": input.UA(),
            "UREA": input.UREA(),
            "age": input.Age(),
        }

        feature_names = list(data.keys())

        orig_data = pd.DataFrame([data])
        # print("Original Data:", data)

        # zero_count = orig_data[orig_data == 0].count().sum()
        # # print("Zero Count:", zero_count)
        #
        # if zero_count > 0:
        #     return None

        others_df = orig_data.drop(["NEUT#", "MXD#", "MXD%"], axis=1)
        others_zero_count = others_df[others_df == 0].count().sum()

        if others_zero_count > 0:
            return None

        scaled_data = pd.DataFrame(scaler.transform(orig_data))
        # print("Standardized Data:", scaled_data)

        shap_explainer = shap.TreeExplainer(model_with_AL)
        shap_values = shap_explainer.shap_values(scaled_data)
        # print("SHAP Values:", shap_values[0, :])

        shap.initjs()
        # plt.figure(figsize = (10, 3))
        shap.force_plot(shap_explainer.expected_value, shap_values[0, :], orig_data,
                        feature_names = feature_names, show = False, matplotlib = True)

app = App(app_ui, server)

if __name__ == "__main__":
    app.run()