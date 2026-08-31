import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path


# ============================================================
# INTERACTIVE SCENARIO COMPARISON TOOL
# CHAPTER 7 - BACHELOR THESIS
# ============================================================

st.set_page_config(
    page_title="Scenario Comparison Tool",
    page_icon="⚡",
    layout="wide"
)


# ============================================================
# 1. INPUT FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DEMAND_FILE = BASE_DIR / "demand_profile.csv"
SOLAR_FILE = BASE_DIR / "solar_profile.csv"
WIND_FILE = BASE_DIR / "wind_profile.csv"


# ============================================================
# 2. LOAD INPUT DATA
# ============================================================

@st.cache_data
def load_profiles():

    demand_df = pd.read_csv(
        DEMAND_FILE
    )

    solar_df = pd.read_csv(
        SOLAR_FILE
    )

    wind_df = pd.read_csv(
        WIND_FILE
    )

    N_HOURS = 8760

    demand = (
        demand_df["demand_MW"]
        .iloc[:N_HOURS]
        .to_numpy(dtype=float)
    )

    solar_norm = (
        solar_df["Solar_Norm"]
        .iloc[:N_HOURS]
        .to_numpy(dtype=float)
    )

    wind_norm = (
        wind_df["Normalized_Wind_Output"]
        .iloc[:N_HOURS]
        .to_numpy(dtype=float)
    )

    if len(demand) != N_HOURS:
        raise ValueError(
            "Demand profile must contain 8760 hours."
        )

    if len(solar_norm) != N_HOURS:
        raise ValueError(
            "Solar profile must contain 8760 hours."
        )

    if len(wind_norm) != N_HOURS:
        raise ValueError(
            "Wind profile must contain 8760 hours."
        )

    return (
        demand,
        solar_norm,
        wind_norm
    )


demand, solar_norm, wind_norm = load_profiles()

N_HOURS = 8760


# ============================================================
# 3. DEMAND
# ============================================================

annual_demand_MWh = demand.sum()

peak_demand_MW = demand.max()


# ============================================================
# 4. REFERENCE INSTALLED CAPACITIES
# ============================================================
#
# Chapter 5 reference:
#
# PV + wind = 93.5% of annual demand
# PV share  = 65%
# Wind share = 35%
#
# These values establish the default installed capacities.
#
# The user can then change the installed capacities directly.
# ============================================================

RE_PENETRATION = 0.935

PV_SHARE = 0.65

WIND_SHARE = 0.35


target_RE_MWh = (
    RE_PENETRATION
    * annual_demand_MWh
)

target_PV_MWh = (
    PV_SHARE
    * target_RE_MWh
)

target_WIND_MWh = (
    WIND_SHARE
    * target_RE_MWh
)


REFERENCE_PV_CAPACITY_MW = (
    target_PV_MWh
    / solar_norm.sum()
)

REFERENCE_WIND_CAPACITY_MW = (
    target_WIND_MWh
    / wind_norm.sum()
)


# ============================================================
# 5. BATTERY EFFICIENCY
# ============================================================

ROUND_TRIP_EFFICIENCY = 0.87

eta_charge = np.sqrt(
    ROUND_TRIP_EFFICIENCY
)

eta_discharge = np.sqrt(
    ROUND_TRIP_EFFICIENCY
)


# ============================================================
# 6. ECONOMIC INPUTS
# ============================================================

PV_CAPEX = 680.0

WIND_CAPEX = 1370.0

BATTERY_CAPEX = 229.0

PV_OM = 22.0

WIND_OM = 44.0

BATTERY_OM_PERCENT = 0.025

GAS_GENERATION_COST = 40.19

CCGT_EFFICIENCY = 0.55

LHV_GAS = 9.8

EXPORT_PRICE = 217.0

EF_GAS = 0.404


# ============================================================
# 7. BATTERY DISPATCH
# ============================================================

def simulate_one_year(
    demand,
    renewable,
    battery_power_MW,
    battery_energy_MWh,
    eta_charge,
    eta_discharge,
    initial_SOC_MWh
):

    SOC = initial_SOC_MWh

    gas_generation = 0.0

    curtailment = 0.0

    battery_charge = 0.0

    battery_discharge = 0.0

    hourly_gas = np.zeros(
        len(demand)
    )

    hourly_SOC = np.zeros(
        len(demand)
    )


    for h, (load, re) in enumerate(
        zip(demand, renewable)
    ):

        # ====================================================
        # RENEWABLE SURPLUS
        # ====================================================

        if re >= load:

            surplus_hour = (
                re - load
            )

            max_charge_by_power = (
                battery_power_MW
            )

            max_charge_by_capacity = max(
                (
                    battery_energy_MWh
                    - SOC
                )
                / eta_charge,
                0.0
            )

            charge_input = min(
                surplus_hour,
                max_charge_by_power,
                max_charge_by_capacity
            )

            charge_input = max(
                charge_input,
                0.0
            )

            SOC += (
                charge_input
                * eta_charge
            )

            battery_charge += (
                charge_input
            )

            curtailment += (
                surplus_hour
                - charge_input
            )


        # ====================================================
        # RENEWABLE DEFICIT
        # ====================================================

        else:

            deficit_hour = (
                load - re
            )

            max_discharge_by_power = (
                battery_power_MW
            )

            max_discharge_by_SOC = (
                SOC
                * eta_discharge
            )

            discharge_to_load = min(
                deficit_hour,
                max_discharge_by_power,
                max_discharge_by_SOC
            )

            discharge_to_load = max(
                discharge_to_load,
                0.0
            )

            SOC -= (
                discharge_to_load
                / eta_discharge
            )

            battery_discharge += (
                discharge_to_load
            )

            gas_hour = (
                deficit_hour
                - discharge_to_load
            )

            gas_generation += (
                gas_hour
            )

            hourly_gas[h] = (
                gas_hour
            )

        hourly_SOC[h] = SOC


    return {

        "gas_MWh":
            gas_generation,

        "curtailment_MWh":
            curtailment,

        "battery_charge_MWh":
            battery_charge,

        "battery_discharge_MWh":
            battery_discharge,

        "final_SOC_MWh":
            SOC,

        "hourly_gas":
            hourly_gas,

        "hourly_SOC":
            hourly_SOC
    }


# ============================================================
# 8. CYCLIC SOC
# ============================================================

def find_cyclic_SOC(
    demand,
    renewable,
    battery_power_MW,
    battery_energy_MWh,
    eta_charge,
    eta_discharge,
    tolerance_MWh=0.001,
    max_iterations=1000
):

    if battery_energy_MWh == 0:

        return 0.0


    SOC_start = (
        0.50
        * battery_energy_MWh
    )


    for _ in range(
        max_iterations
    ):

        result = simulate_one_year(

            demand=demand,

            renewable=renewable,

            battery_power_MW=battery_power_MW,

            battery_energy_MWh=battery_energy_MWh,

            eta_charge=eta_charge,

            eta_discharge=eta_discharge,

            initial_SOC_MWh=SOC_start
        )

        SOC_end = (
            result["final_SOC_MWh"]
        )

        difference = (
            SOC_end
            - SOC_start
        )


        if abs(
            difference
        ) <= tolerance_MWh:

            return SOC_start


        SOC_start = SOC_end


    return SOC_start


# ============================================================
# 9. ECONOMIC CALCULATION
# ============================================================

def calculate_economics(
    pv_capacity_MW,
    wind_capacity_MW,
    battery_energy_MWh,
    gas_MWh
):

    # --------------------------------------------------------
    # PV CAPEX
    # --------------------------------------------------------

    pv_capex = (
        pv_capacity_MW
        * 1000
        * PV_CAPEX
    )


    # --------------------------------------------------------
    # WIND CAPEX
    # --------------------------------------------------------

    wind_capex = (
        wind_capacity_MW
        * 1000
        * WIND_CAPEX
    )


    # --------------------------------------------------------
    # BATTERY CAPEX
    # --------------------------------------------------------

    battery_capex = (
        battery_energy_MWh
        * 1000
        * BATTERY_CAPEX
    )


    # --------------------------------------------------------
    # TOTAL CAPEX
    # --------------------------------------------------------

    total_capex = (
        pv_capex
        + wind_capex
        + battery_capex
    )


    # --------------------------------------------------------
    # PV O&M
    # --------------------------------------------------------

    pv_om = (
        pv_capacity_MW
        * 1000
        * PV_OM
    )


    # --------------------------------------------------------
    # WIND O&M
    # --------------------------------------------------------

    wind_om = (
        wind_capacity_MW
        * 1000
        * WIND_OM
    )


    # --------------------------------------------------------
    # BATTERY O&M
    # --------------------------------------------------------

    battery_om = (
        battery_capex
        * BATTERY_OM_PERCENT
    )


    total_future_om = (
        pv_om
        + wind_om
        + battery_om
    )


    # --------------------------------------------------------
    # CURRENT GAS SYSTEM COST
    # --------------------------------------------------------

    current_system_cost = (
        annual_demand_MWh
        * GAS_GENERATION_COST
    )


    # --------------------------------------------------------
    # FUTURE GAS GENERATION COST
    # --------------------------------------------------------

    future_gas_cost = (
        gas_MWh
        * GAS_GENERATION_COST
    )


    # --------------------------------------------------------
    # FUTURE SYSTEM COST
    # --------------------------------------------------------

    future_system_cost = (
        future_gas_cost
        + total_future_om
    )


    # --------------------------------------------------------
    # ANNUAL OPERATING COST SAVING
    # --------------------------------------------------------

    annual_saving = (
        current_system_cost
        - future_system_cost
    )


    # --------------------------------------------------------
    # GAS SAVED
    # --------------------------------------------------------

    gas_saved_twh = (

        (
            annual_demand_MWh
            - gas_MWh
        )
        / 1_000_000
    )


    # --------------------------------------------------------
    # GAS FUEL ENERGY
    # --------------------------------------------------------

    fuel_energy_twh = (

        gas_saved_twh
        / CCGT_EFFICIENCY
    )


    fuel_energy_kwh = (

        fuel_energy_twh
        * 1_000_000_000
    )


    # --------------------------------------------------------
    # GAS VOLUME
    # --------------------------------------------------------

    gas_volume_m3 = (

        fuel_energy_kwh
        / LHV_GAS
    )


    gas_volume_1000m3 = (

        gas_volume_m3
        / 1000
    )


    # --------------------------------------------------------
    # GAS EXPORT REVENUE
    # --------------------------------------------------------

    export_revenue = (

        gas_volume_1000m3
        * EXPORT_PRICE
    )


    # --------------------------------------------------------
    # ANNUAL NET ECONOMIC BENEFIT
    # --------------------------------------------------------

    annual_net_benefit = (

        annual_saving
        + export_revenue
    )


    # --------------------------------------------------------
    # SIMPLE PAYBACK
    # --------------------------------------------------------

    if annual_net_benefit > 0:

        payback_years = (

            total_capex
            / annual_net_benefit
        )

    else:

        payback_years = np.nan


    return {

        "pv_capex":
            pv_capex,

        "wind_capex":
            wind_capex,

        "battery_capex":
            battery_capex,

        "total_capex":
            total_capex,

        "total_future_om":
            total_future_om,

        "future_gas_cost":
            future_gas_cost,

        "annual_saving":
            annual_saving,

        "export_revenue":
            export_revenue,

        "annual_net_benefit":
            annual_net_benefit,

        "payback_years":
            payback_years
    }


# ============================================================
# 10. PAGE TITLE
# ============================================================

st.title(
    "Interactive Scenario Comparison Tool"
)

st.write(
    "Compare different installed PV, wind and "
    "battery-storage configurations."
)


# ============================================================
# 11. SIDEBAR - SCENARIO
# ============================================================

st.sidebar.header(
    "Scenario Selection"
)


scenario_type = st.sidebar.radio(

    "Scenario type",

    [
        "No renewable generation",
        "Installed-capacity scenario"
    ]
)


# ============================================================
# 12. PV AND WIND CAPACITIES
# ============================================================

if (
    scenario_type
    == "No renewable generation"
):

    pv_capacity_MW = 0.0

    wind_capacity_MW = 0.0

else:

    pv_capacity_GW = st.sidebar.number_input(

        "Installed PV capacity (GW)",

        min_value=0.0,

        max_value=30.0,

        value=11.429,

        step=0.001,

        format="%.3f"
    )


    wind_capacity_GW = st.sidebar.number_input(

        "Installed wind capacity (GW)",

        min_value=0.0,

        max_value=30.0,

        value=10.068,

        step=0.001,

        format="%.3f"
    )


    pv_capacity_MW = (
        pv_capacity_GW
        * 1000
    )


    wind_capacity_MW = (
        wind_capacity_GW
        * 1000
    )


# ============================================================
# 13. BATTERY PARAMETERS
# ============================================================

st.sidebar.header(
    "Battery Parameters"
)


battery_power_percent = st.sidebar.slider(

    "Battery power (% of peak demand)",

    min_value=0,

    max_value=100,

    value=50,

    step=1
)


duration_hours = st.sidebar.select_slider(

    "Storage duration (hours)",

    options=[
        2,
        4,
        6,
        8
    ],

    value=8
)


# ============================================================
# 14. HOURLY RENEWABLE GENERATION
# ============================================================

pv_generation = (

    solar_norm
    * pv_capacity_MW
)


wind_generation = (

    wind_norm
    * wind_capacity_MW
)


renewable_generation = (

    pv_generation
    + wind_generation
)


# ============================================================
# 15. BATTERY SIZE
# ============================================================

battery_power_MW = (

    battery_power_percent
    / 100
    * peak_demand_MW
)


battery_energy_MWh = (

    battery_power_MW
    * duration_hours
)


# ============================================================
# 16. TECHNICAL SIMULATION
# ============================================================

if (

    pv_capacity_MW == 0
    and
    wind_capacity_MW == 0
):

    gas_MWh = (
        annual_demand_MWh
    )

    curtailment_MWh = 0.0

    battery_charge_MWh = 0.0

    battery_discharge_MWh = 0.0

    initial_SOC_MWh = 0.0

    hourly_gas = (
        demand.copy()
    )

    hourly_SOC = np.zeros(
        N_HOURS
    )

    battery_power_MW = 0.0

    battery_energy_MWh = 0.0


else:

    # --------------------------------------------------------
    # Find cyclic SOC
    # --------------------------------------------------------

    initial_SOC_MWh = find_cyclic_SOC(

        demand=demand,

        renewable=renewable_generation,

        battery_power_MW=battery_power_MW,

        battery_energy_MWh=battery_energy_MWh,

        eta_charge=eta_charge,

        eta_discharge=eta_discharge
    )


    # --------------------------------------------------------
    # Final annual simulation
    # --------------------------------------------------------

    result = simulate_one_year(

        demand=demand,

        renewable=renewable_generation,

        battery_power_MW=battery_power_MW,

        battery_energy_MWh=battery_energy_MWh,

        eta_charge=eta_charge,

        eta_discharge=eta_discharge,

        initial_SOC_MWh=initial_SOC_MWh
    )


    gas_MWh = (
        result["gas_MWh"]
    )

    curtailment_MWh = (
        result["curtailment_MWh"]
    )

    battery_charge_MWh = (
        result["battery_charge_MWh"]
    )

    battery_discharge_MWh = (
        result["battery_discharge_MWh"]
    )

    hourly_gas = (
        result["hourly_gas"]
    )

    hourly_SOC = (
        result["hourly_SOC"]
    )


# ============================================================
# 17. ANNUAL ENERGY BALANCE
# ============================================================
#
# Total PV + wind generation is determined only by the
# installed PV and wind capacities.
#
# Battery size does NOT change total renewable generation.
#
# The battery changes how much renewable electricity can be
# shifted from surplus hours to deficit hours.
#
# Renewable generation is divided into:
#
#   Direct renewable supply to demand
#   Battery charging
#   Curtailment
#
# The battery discharge then contributes to meeting demand.
# ============================================================

total_pv_generation_MWh = (
    pv_generation.sum()
)

total_wind_generation_MWh = (
    wind_generation.sum()
)

total_renewable_generation_MWh = (
    total_pv_generation_MWh
    + total_wind_generation_MWh
)


# Renewable electricity delivered to demand
# = demand not supplied by gas.

renewable_delivered_MWh = (

    annual_demand_MWh
    - gas_MWh
)


# Direct renewable electricity used by demand
# excludes electricity first sent into the battery.

direct_renewable_MWh = (

    renewable_delivered_MWh
    - battery_discharge_MWh
)


# The small numerical protection prevents tiny floating-point
# negative values from appearing.

direct_renewable_MWh = max(
    direct_renewable_MWh,
    0.0
)


# Renewable energy accounting check:
#
# Total renewable generation =
# direct renewable use
# + battery charging input
# + curtailment

energy_balance_difference_MWh = (

    total_renewable_generation_MWh
    - (
        direct_renewable_MWh
        + battery_charge_MWh
        + curtailment_MWh
    )
)


# Renewable generation percentages are intentionally NOT
# displayed because they were removed from the website.

total_pv_generation_TWh = (
    total_pv_generation_MWh
    / 1_000_000
)

total_wind_generation_TWh = (
    total_wind_generation_MWh
    / 1_000_000
)

total_renewable_generation_TWh = (
    total_renewable_generation_MWh
    / 1_000_000
)

renewable_delivered_TWh = (
    renewable_delivered_MWh
    / 1_000_000
)

direct_renewable_TWh = (
    direct_renewable_MWh
    / 1_000_000
)

battery_charge_TWh = (
    battery_charge_MWh
    / 1_000_000
)

battery_discharge_TWh = (
    battery_discharge_MWh
    / 1_000_000
)

curtailment_TWh = (
    curtailment_MWh
    / 1_000_000
)

gas_generation_TWh = (
    gas_MWh
    / 1_000_000
)


# ============================================================
# 18. ECONOMIC CALCULATION
# ============================================================

economics = calculate_economics(

    pv_capacity_MW=
        pv_capacity_MW,

    wind_capacity_MW=
        wind_capacity_MW,

    battery_energy_MWh=
        battery_energy_MWh,

    gas_MWh=
        gas_MWh
)


# ============================================================
# 19. CO2 CALCULATION
# ============================================================

baseline_CO2_Mt = (

    annual_demand_MWh
    * EF_GAS
    / 1_000_000
)


scenario_CO2_Mt = (

    gas_MWh
    * EF_GAS
    / 1_000_000
)


avoided_CO2_Mt = (

    baseline_CO2_Mt
    - scenario_CO2_Mt
)


CO2_reduction_percent = (

    avoided_CO2_Mt
    / baseline_CO2_Mt
    * 100
)


# ============================================================
# 20. ANNUAL ENERGY SUMMARY
# ============================================================

st.header(
    "Annual Energy Summary"
)


energy_col1, energy_col2, energy_col3 = st.columns(3)


with energy_col1:

    st.metric(

        "PV generation",

        f"{total_pv_generation_TWh:.2f} TWh/year"
    )


with energy_col2:

    st.metric(

        "Wind generation",

        f"{total_wind_generation_TWh:.2f} TWh/year"
    )


with energy_col3:

    st.metric(

        "Total PV + wind generation",

        f"{total_renewable_generation_TWh:.2f} TWh/year"
    )


energy_col4, energy_col5, energy_col6 = st.columns(3)


with energy_col4:

    st.metric(

        "Renewable electricity delivered",

        f"{renewable_delivered_TWh:.2f} TWh/year"
    )


with energy_col5:

    st.metric(

        "Gas generation",

        f"{gas_generation_TWh:.2f} TWh/year"
    )


with energy_col6:

    st.metric(

        "Renewable energy curtailed",

        f"{curtailment_TWh:.2f} TWh/year"
    )


# ============================================================
# 21. ADDITIONAL ENERGY INFORMATION
# ============================================================


# ============================================================
# 22. GENERATION GRAPH
# ============================================================

st.header(
    "Electricity Generation and Demand"
)


profile_period = st.radio(

    "Select profile period",

    [
        "Month",
        "Week"
    ],

    horizontal=True
)


# ============================================================
# MONTH SELECTION
# ============================================================

if profile_period == "Month":

    month_names = [

        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December"
    ]


    days_per_month = [

        31,
        28,
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31
    ]


    selected_month = st.selectbox(

        "Select month",

        month_names,

        index=6
    )


    month_index = (
        month_names.index(
            selected_month
        )
    )


    start_hour = (

        sum(
            days_per_month[
                :month_index
            ]
        )
        * 24
    )


    end_hour = (

        start_hour
        + days_per_month[
            month_index
        ]
        * 24
    )


    graph_label = selected_month


# ============================================================
# WEEK SELECTION
# ============================================================

else:

    selected_week = st.selectbox(

        "Select week",

        list(range(1, 53)),

        index=26,

        format_func=lambda x:
            f"Week {x}"
    )


    start_hour = (

        (selected_week - 1)
        * 7
        * 24
    )


    end_hour = min(

        start_hour
        + 7 * 24,

        N_HOURS
    )


    graph_label = (
        f"Week {selected_week}"
    )


# ============================================================
# GRAPH DATA
# ============================================================

demand_plot = demand[
    start_hour:end_hour
]

pv_plot = pv_generation[
    start_hour:end_hour
]

wind_plot = wind_generation[
    start_hour:end_hour
]

gas_plot = hourly_gas[
    start_hour:end_hour
]


x_values = np.arange(

    1,

    len(demand_plot) + 1
)


# ============================================================
# GENERATION GRAPH
# ============================================================

fig_generation = go.Figure()


fig_generation.add_trace(

    go.Scatter(

        x=x_values,

        y=pv_plot,

        mode="lines",

        name="PV",

        line=dict(

            color="#E69F00",

            width=2.5
        )
    )
)


fig_generation.add_trace(

    go.Scatter(

        x=x_values,

        y=wind_plot,

        mode="lines",

        name="Wind",

        line=dict(

            color="#0072B2",

            width=2.5
        )
    )
)


fig_generation.add_trace(

    go.Scatter(

        x=x_values,

        y=gas_plot,

        mode="lines",

        name="Gas",

        line=dict(

            color="#7B2CBF",

            width=2.5
        )
    )
)


fig_generation.add_trace(

    go.Scatter(

        x=x_values,

        y=demand_plot,

        mode="lines",

        name="Demand",

        line=dict(

            color="#222222",

            width=3.5
        )
    )
)


fig_generation.update_layout(

    title=dict(

        text=(
            "Hourly Electricity Generation "
            f"and Demand — {graph_label}"
        ),

        x=0.5,

        xanchor="center"
    ),

    xaxis=dict(

        title=(
            "Hour of "
            + (
                "month"
                if profile_period == "Month"
                else "week"
            )
        ),

        showgrid=True,

        gridcolor="#E6E6E6",

        zeroline=False
    ),

    yaxis=dict(

        title="Power (MW)",

        showgrid=True,

        gridcolor="#E6E6E6",

        zeroline=False
    ),

    height=560,

    hovermode="x unified",

    plot_bgcolor="white",

    paper_bgcolor="white",

    margin=dict(

        l=70,

        r=30,

        t=80,

        b=70
    ),

    legend=dict(

        orientation="h",

        yanchor="bottom",

        y=1.02,

        xanchor="center",

        x=0.5,

        bgcolor="rgba(255,255,255,0.9)"
    )
)


st.plotly_chart(

    fig_generation,

    use_container_width=True
)


# ============================================================
# 23. CARBON EMISSIONS
# ============================================================

st.header(
    "Carbon Emissions and Avoided Emissions"
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(

        "Gas-only emissions",

        f"{baseline_CO2_Mt:.2f} MtCO₂/year"
    )


with col2:

    st.metric(

        "Scenario emissions",

        f"{scenario_CO2_Mt:.2f} MtCO₂/year"
    )


with col3:

    st.metric(

        "Avoided emissions",

        f"{avoided_CO2_Mt:.2f} MtCO₂/year",

        f"{CO2_reduction_percent:.1f}% reduction"
    )


# ============================================================
# 24. CO2 GRAPH
# ============================================================

fig_carbon = go.Figure()


fig_carbon.add_trace(

    go.Bar(

        x=[
            "Gas-only reference",
            "Selected scenario"
        ],

        y=[
            baseline_CO2_Mt,
            scenario_CO2_Mt
        ],

        marker_color="#009E73"
    )
)


fig_carbon.update_layout(

    yaxis_title="MtCO₂/year",

    height=400,

    showlegend=False,

    plot_bgcolor="white",

    paper_bgcolor="white"
)


st.plotly_chart(

    fig_carbon,

    use_container_width=True
)


# ============================================================
# 25. INVESTMENT COSTS
# ============================================================

st.header(
    "Investment Costs"
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(

        "PV",

        f"${economics['pv_capex'] / 1e9:.2f} bn"
    )


with col2:

    st.metric(

        "Wind",

        f"${economics['wind_capex'] / 1e9:.2f} bn"
    )


with col3:

    st.metric(

        "Battery",

        f"${economics['battery_capex'] / 1e9:.2f} bn"
    )


with col4:

    st.metric(

        "Total",

        f"${economics['total_capex'] / 1e9:.2f} bn"
    )


# ============================================================
# 26. COST GRAPH
# ============================================================

fig_cost = go.Figure()


fig_cost.add_trace(

    go.Bar(

        x=[
            "PV",
            "Wind",
            "Battery"
        ],

        y=[

            economics["pv_capex"]
            / 1e9,

            economics["wind_capex"]
            / 1e9,

            economics["battery_capex"]
            / 1e9
        ],

        marker_color="#56B4E9"
    )
)


fig_cost.update_layout(

    yaxis_title=(
        "Investment cost (billion USD)"
    ),

    height=400,

    showlegend=False,

    plot_bgcolor="white",

    paper_bgcolor="white"
)


st.plotly_chart(

    fig_cost,

    use_container_width=True
)


# ============================================================
# 27. PAYBACK
# ============================================================

st.header(
    "Estimated Payback Period"
)


def truncate_one_decimal(value):

    if not np.isfinite(value):

        return "N/A"

    return (
        f"{np.floor(value * 10) / 10:.1f}"
    )


if np.isfinite(
    economics["payback_years"]
):

    st.metric(

        "Estimated simple payback",

        (
            f"≈ {truncate_one_decimal(economics['payback_years'])} "
            "years"
        )
    )

else:

    st.info(

        "A positive annual net economic benefit "
        "is not obtained for this scenario. "
        "A simple payback period cannot be estimated."
    )

    # ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <hr>
    <div style="text-align: center; color: #666; font-size: 14px;">
        <strong>Developed by Obada Kurdi</strong><br>
        Bachelor Thesis — Assessment of Solar–Wind Integration and Battery Storage
        in Turkmenistan’s Future Power System<br>
        © 2026 Obada Kurdi. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)