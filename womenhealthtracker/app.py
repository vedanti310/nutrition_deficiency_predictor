import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import openai
from streamlit_lottie import st_lottie
import requests
from datetime import date, timedelta
import plotly.graph_objects as go
import calendar

# ===== Load Dataset and Train Model =====
df = pd.read_csv("deficency2.5.csv")
X = df.drop(['Name', 'Deficiency'], axis=1)
y = df['Deficiency']

label_encoders = {}
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
target_le = LabelEncoder()
y_encoded = target_le.fit_transform(y)

model = DecisionTreeClassifier()
model.fit(X, y_encoded)


# ===== Lottie Loader =====
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"Lottie load error: {e}")
    return None


lottie_health = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_mjlh3hcy.json")
lottie_period = load_lottieurl("https://lottie.host/f7ac389c-e116-4bcb-a9fa-2d3d014ff2a3/5bqW0AOlY1.json")


# ===== Suggestions Dictionary =====
tips_dict = {
    "Iron": "🥩 Eat iron-rich foods like red meat, spinach, lentils, and beans.\n🧬 Include vitamin C-rich foods.\n🩺 Consider supplements.",
    "Vitamin_D": "☀️ Get sunlight exposure daily.\n🥛 Eat fortified dairy and fatty fish.\n🩺 Supplements if needed.",
    "Calcium": "🥛 Consume dairy, leafy greens, almonds, and tofu.\n🏋️‍♀️ Do weight-bearing exercises.\n🩺 Consult your doctor.",
    "Vitamin_B12": "🍖 Eat meat, fish, dairy, and eggs.\n🌱 Vegetarians: fortified foods or supplements.\n🩺 Ask your doctor about injections.",
    "Folate": "🥦 Eat leafy greens, nuts, beans, and cereals.\n👩‍⚕️ Important during pregnancy.\n🩺 May require supplements.",
    "Vitamin_C": "🍊 Eat citrus, strawberries, bell peppers, and broccoli.\n🩺 Supports immune system and healing.",
    "Magnesium": "🥑 Eat nuts, seeds, grains, and leafy greens.\n🧘‍♀️ Helps nerves and muscles.",
    "Zinc": "🥩 Eat meat, legumes, and seeds.\n🩺 Good for immune system and wound healing.",
    "Vitamin_B6": "🍌 Eat bananas, potatoes, poultry, and fish.\n🧠 Supports brain health and metabolism.",
    "Vitamin_E": "🌰 Eat nuts, seeds, and leafy vegetables.\n🩺 Acts as antioxidant and immune booster."
}

# ===== UI Layout =====
st.set_page_config(page_title="Women's Health App", layout="wide")
st.markdown(
    """
    <style>

body {
    background: linear-gradient(135deg, #fce4ec, #f3e5f5) !important;
    min-height: 100vh;
}

.css-18e3th9 {
    background: linear-gradient(135deg, #fce4ec, #f3e5f5);
    color: #4a235a;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Style for main header */
h1, h2, h3 {
    color: #7b2e68;
}
/* h1 {
        text-align: center !important;
    } */

/* Style for buttons */
.stButton > button {
    background-color: #d6336c;
    color: white;
    border-radius: 12px;
    padding: 8px 20px;
    font-weight: bold;
    border: none;
    transition: background-color 0.3s ease;
}

.stButton > button:hover {
    background-color: #a5274b;
}

/* Style for input fields */
.stTextInput>div>input,
.stNumberInput>div>input,
.stSelectbox>div>div>div>select {
    border: 2px solid #d6336c;
    border-radius: 10px;
    padding: 6px 10px;
    color: #4a235a;
}

/* Multiselect box style */
.stMultiSelect>div>div>div {
    border-radius: 10px;
    border: 2px solid #d6336c;
}

/* Sidebar (if any) */
.css-1d391kg {
    background: linear-gradient(135deg, #fce4ec, #f3e5f5);
}

/* Streamlit header and footer removal for clean look */
.css-1v3fvcr, .css-1v3fvcr + div {
    display: none;
}
/* Strong selector for main content */
    .css-18e3th9 {
        background: linear-gradient(135deg, #fce4ec, #f3e5f5) !important;
        color: #4a235a !important;
        min-height: 100vh;
    }

    /* Optional: also the app container */
    .css-1d391kg {
        background: linear-gradient(135deg, #fce4ec, #f3e5f5) !important;
    }
</style>

    """,
    unsafe_allow_html=True
)


st.title("🌸 Women's Health & Wellness App")

tabs = st.tabs(["🥗 Nutrition Predictor", "🩸 Period Tracker", "💬 Chatbot"])

# ===== Tab 1: Nutrition Predictor =====
with tabs[0]:
    st.header("🥗 Nutrition Deficiency Predictor")
    if lottie_health:
        st_lottie(lottie_health, height=200)
    else:
        st.info("Animation not loaded.")

    st.subheader("Fill your details")

    user_input = {}
    user_name = st.text_input("Enter your name")
    col1, col2, col3 = st.columns(3)
    user_input['Age'] = col1.number_input("Age", min_value=0, max_value=120, step=1)
    user_input['Gender'] = col2.selectbox("Gender", ['Male', 'Female'])
    user_input['Blood_Type'] = col3.selectbox("Blood Type", ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'])

    binary_cols = [col for col in X.columns if col not in ['Age', 'Gender', 'Blood_Type']]
    selected_symptoms = st.multiselect("Select symptoms/conditions you have:", options=binary_cols)
    for col in binary_cols:
        user_input[col] = 'Yes' if col in selected_symptoms else 'No'

    if st.button("🔍 Predict"):
        if not selected_symptoms:
            st.warning("⚠️ Please select symptoms for accurate prediction.")
        else:
            input_df = pd.DataFrame([user_input], columns=X.columns)
            # Encode categorical inputs
            for col, le in label_encoders.items():
                input_df[col] = le.transform(input_df[col])
            prediction = model.predict(input_df)
            predicted_label = target_le.inverse_transform(prediction)[0].strip()

            st.success(f"Hello {user_name}! 🔍 Likely Deficiency: **{predicted_label}**")
            health_score = max(0, 100 - (len(selected_symptoms) * 5))
            st.metric("🩺 Estimated Health Score", f"{health_score}/100")
            st.progress(health_score / 100)

            if predicted_label in tips_dict:
                st.info("💡 **Suggestions & Tips:**")
                st.markdown(tips_dict[predicted_label].replace("\n", "<br>"), unsafe_allow_html=True)
            else:
                st.warning(f"No suggestions available for: {predicted_label}")

# ===== Tab 2: Period Tracker =====
from datetime import date, timedelta
import calendar
import plotly.graph_objects as go
import streamlit as st

with tabs[1]:
    st.markdown("<h2 style='text-align: center;'>🩸 Period Tracker</h2>", unsafe_allow_html=True)

    # Inputs
    last_period = st.date_input("Select your last period date", value=date.today())
    cycle_length = st.slider("Average cycle length (days)", 21, 35, 28)
    period_length = st.slider("Period length (days)", 3, 10, 5)

    selected_year = date.today().year
    selected_month = date.today().month

    # Calculate key dates
    ovulation_day = last_period + timedelta(days=cycle_length - 14)
    fertile_start = ovulation_day - timedelta(days=5)
    fertile_end = ovulation_day - timedelta(days=1)
    next_period = last_period + timedelta(days=cycle_length)

    # Build calendar for selected month
    cal = calendar.Calendar(firstweekday=6)  # Start on Sunday
    month_days = list(cal.itermonthdates(selected_year, selected_month))

    weeks = []
    week = []

    for i, day in enumerate(month_days):
        label = str(day.day)
        color = "#ffffff"  # Default white

        # Only label if the day is in selected month
        if day.month == selected_month:
            if last_period <= day < last_period + timedelta(days=period_length):
                label += "\nPeriod"
                color = "#f4a6b1"
            elif fertile_start <= day <= fertile_end:
                label += "\nFertile"
                color = "#ffd1dc"
            elif day == ovulation_day:
                label += "\nOvulation"
                color = "#ff85a2"
            elif day == next_period:
                label += "\nNext"
                color = "#ffb3c6"

        week.append({'label': label if day.month == selected_month else '', 'color': color if day.month == selected_month else '#ffffff'})

        if (i + 1) % 7 == 0:
            weeks.append(week)
            week = []

    if week:
        while len(week) < 7:
            week.append({'label': '', 'color': '#ffffff'})
        weeks.append(week)

    # Prepare data for Plotly table
    cell_values = list(zip(*[[day['label'] for day in week] for week in weeks]))
    cell_colors = list(zip(*[[day['color'] for day in week] for week in weeks]))

    # Plot
    header_days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    fig = go.Figure(data=[go.Table(
        header=dict(values=header_days, fill_color='#a83250', font=dict(color='white', size=14), align='center'),
        cells=dict(
    values=cell_values,
    fill_color=cell_colors,
    align='center',
    font=dict(color='black', size=14),
    height=80,  # ← Increase from 60 to 80 (or more)
    line_color='darkgrey'
)
)
    ])
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=600)


    st.plotly_chart(fig, use_container_width=True)


# ===== Tab 3: Chatbot =====
with tabs[2]:
    st.header("💬 Chat with Health Assistant")

    if "OPENAI_API_KEY" not in st.secrets:
        st.warning("OpenAI API key not found. Please add it to Streamlit secrets.")
    else:
        openai.api_key = st.secrets["OPENAI_API_KEY"]

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                {"role": "system", "content": "You are a helpful AI health assistant. Answer general health and nutrition questions helpfully."}
            ]

        def generate_response():
            user_query = st.session_state.user_input.strip()
            if user_query:
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=st.session_state.chat_history,
                        temperature=0.7,
                        max_tokens=500
                    )
                    reply = response.choices[0].message["content"]
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                except Exception as e:
                    st.session_state.chat_history.append({"role": "assistant", "content": f"⚠️ Error: {e}"})
                st.session_state.user_input = ""

        # Display chat history
        for msg in st.session_state.chat_history[1:]:
            role = "You" if msg["role"] == "user" else "Health Bot"
            st.markdown(f"**{role}:** {msg['content']}")

        st.text_input("Type your question and press Enter:", key="user_input", on_change=generate_response)
        st.caption("🧠 Powered by OpenAI GPT-3.5 Turbo")
