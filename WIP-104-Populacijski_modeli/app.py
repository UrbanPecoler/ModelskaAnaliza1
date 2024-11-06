import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
import io

def epidemic_model(D0, B0, I0, alpha, beta, time_points):
    def diff_eq(V, t, alpha, beta):
        Y = np.zeros(3)
        Y[0] = -alpha * V[0] * V[1]
        Y[1] = alpha * V[0] * V[1] - beta * V[1]
        Y[2] = beta * V[1]
        return Y
    
    initial_conditions = [D0, B0, I0]
    result = odeint(diff_eq, initial_conditions, time_points, args=(alpha, beta))
    return result


def lotka_volterra(t, y, alpha, beta, gamma, delta):
    Z, L = y
    dZdt = alpha * Z - beta * Z * L
    dLdt = -gamma * L + delta * Z * L
    return [dZdt, dLdt]

def brezdim_l_v(t, y, p):
    z, l = y
    dldt = p*z*(1-l)
    dzdt = (l/p)*(z-1)
    return [dzdt, dldt]


# app.py
st.set_page_config(layout="wide")  

tab1, tab2 = st.tabs(["EPIDEMIJA", "LOTKA-VOLTERRA"])

with tab1:
    time_points = np.linspace(0, 100, 200)

    col1, col2 = st.columns([2,3])
    with col1:
        st.write("")  
        st.markdown("***") 
        st.write("")
        D0 = st.slider("D0 (Susceptible)", 0.0, 1.0, 0.5, 0.01)
        B0 = st.slider("B0 (Infected)", 0.0, 1.0 - D0, 0.1, 0.01)  # Ensure B0 max is 1-D0
        alpha = st.slider("Alpha (Infection Rate)", 0.0, 5.0, 0.3, 0.01)
        beta = st.slider("Beta (Recovery Rate)", 0.0, 5.0, 0.4, 0.01)
        st.write("")  
        st.markdown("***") 

    I0 = 1 - D0 - B0

    result = epidemic_model(D0, B0, I0, alpha, beta, time_points)

    with col2:
        fig, ax = plt.subplots()
        ax.plot(time_points, result[:, 0], label="Dovzetni", color="blue")
        ax.plot(time_points, result[:, 1], label="Bolni", color="red")
        ax.plot(time_points, result[:, 2], label="Imuni", color="green")
        ax.set_xlabel("Čas")
        ax.set_ylabel("Procent populacije")
        ax.legend()

        st.pyplot(fig)

    _, col2, _ = st.columns(3)
    # Save plot
    with col2:
        st.write("")
        st.write("")

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        st.download_button(
            label="Download Plot",
            data=buf,
            file_name=f"epdmc_D0-{D0}_B0-{B0}.png",
            mime="image/png"
        )


with tab2:
    st.text("LOTKA - VOLTERRA")

    st.markdown("***")
    st.markdown("***")
    col1, col2 = st.columns([2, 3])
    with col1:
        Z0 = st.slider("Z0 (Zajci)", 0.0, 50.0, 40.0, 1.0)
        L0 = st.slider("L0 (Lisice)", 0.0, 50.0, 9.0, 1.0)  
        alpha = st.slider("Alpha", 0.0, 2.0, 1.0, 0.01)
        beta = st.slider("Beta", 0.0, 1.0, 0.1, 0.01)
        gamma = st.slider("Gamma", 0.0, 2.0, 1.5, 0.01)
        delta = st.slider("Delta", 0.0, 1.0, 0.07, 0.01)


    with col2:
        t_span = (0, 200)
        t_eval = np.linspace(t_span[0], t_span[1], 1000)

        solution = solve_ivp(lotka_volterra, t_span, [Z0, L0], t_eval=t_eval, args=(alpha, beta, gamma, delta))
   
        Z = solution.y[0]
        L = solution.y[1]
        t = solution.t

        fig, ax = plt.subplots()
        ax.plot(Z, L, '-b', label="Fazni diagram")
        ax.set_xlabel("Populacija zajcev (Z)")
        ax.set_ylabel("Populacija lisic (L)")
        ax.set_title("Fazni diagram populacij zajcev in lisic")
        ax.legend()
        st.pyplot(fig)


        t_span = (0, 30)
        t_eval = np.linspace(t_span[0], t_span[1], 1000)

        solution = solve_ivp(lotka_volterra, t_span, [Z0, L0], t_eval=t_eval, args=(alpha, beta, gamma, delta))
        Z = solution.y[0]
        L = solution.y[1]
        t = solution.t

        fig, ax = plt.subplots()

        ax.plot(t, Z, '-g', label="Zajci (Z)")
        ax.plot(t, L, '-r', label="Lisice (L)")
        ax.set_xlabel("Čas")
        ax.set_ylabel("Populacija")
        ax.set_title("Časovni razvoj populacij zajcev in lisic")
        ax.legend()

        st.pyplot(fig)
