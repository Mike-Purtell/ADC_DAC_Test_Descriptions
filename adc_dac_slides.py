import marimo

__generated_with = "0.23.9"
app = marimo.App(
    width="medium",
    layout_file="layouts/adc_dac_slides.slides.json",
)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    import plotly.express as px
    import plotly.io as pio

    # Use a consistent dark theme across all Plotly figures.
    pio.templates.default = "plotly_dark"
    return mo, np, pl, px


@app.cell
def _(mo):
    mo.md("""
    # DAC/ADC Presentation Notebook
    Use the slide deck below to walk through four test sections.

    ## 1. Ideal DAC Behavior and LSB Definition
    A DAC with **N bits** has $2^N$ possible digital codes. The ideal analog output is:

    $V_{ideal}(k) = V_{offset} + k \cdot LSB$

    Where:

    - **LSB (Least Significant Bit)** is the ideal step size:
      $LSB =
    rac{V_{FS} - V_{offset}}{2^N - 1}$
    - $k$ is the digital input code ($0$ to $2^N - 1$)
    - $V_{FS}$ is the full-scale output voltage
    - $V_{offset}$ is the output at code $0$

    All linearity metrics are typically expressed in **LSB units**, making them independent of the DAC's absolute voltage range.
    """)
    return


@app.cell
def _(mo):
    # Page 1 controls
    p1_vmin = mo.ui.text(
        value="0.0",
        label="Min output voltage (V)",
    )
    p1_vmax = mo.ui.text(
        value="5.0",
        label="Max output voltage (V)",
    )
    p1_resolution = mo.ui.dropdown(
        options={"8-bit": 8, "10-bit": 10, "12-bit": 12, "14-bit": 14},
        value='8-bit',
        label="DAC resolution",
    )
    p1_bow = mo.ui.number(
        start=-1.0,
        stop=1.0,
        step=0.001,
        value=0.2,
        label="Simulated bow (V at mid-scale)",
    )
    p1_noise = mo.ui.number(
        start=0.0,
        stop=1.0,
        step=0.0001,
        value=0.02,
        label="Gaussian noise sigma (V)",
    )
    p1_fit_mode = mo.ui.radio(
        options=["End-point based", "Best fit"],
        value="Best fit",
        label="Reference line mode",
    )
    p1_seed = mo.ui.number(
        start=0,
        stop=999999,
        step=1,
        value=42,
        label="Random seed",
    )

    p1_controls = mo.vstack(
        [
            mo.md("## Slide 2: DAC Transfer Curve"),
            mo.hstack([p1_vmin, p1_vmax, p1_resolution]),
            mo.hstack([p1_bow, p1_noise, p1_seed]),
            p1_fit_mode,
        ]
    )
    return (
        p1_bow,
        p1_controls,
        p1_fit_mode,
        p1_noise,
        p1_resolution,
        p1_seed,
        p1_vmax,
        p1_vmin,
    )


@app.cell
def _(
    np,
    p1_bow,
    p1_fit_mode,
    p1_noise,
    p1_resolution,
    p1_seed,
    p1_vmax,
    p1_vmin,
    pl,
    px,
):
    # Page 1 calculations and plots
    try:
        vmin_1 = float(p1_vmin.value)
    except ValueError:
        vmin_1 = 0.0

    try:
        vmax_1 = float(p1_vmax.value)
    except ValueError:
        vmax_1 = 5.0
    bits_1 = int(p1_resolution.value)
    bow_1 = float(p1_bow.value)
    noise_1 = float(p1_noise.value)
    fit_mode_1 = p1_fit_mode.value

    if vmax_1 <= vmin_1:
        vmax_1 = vmin_1 + 1e-6

    max_code_1 = 2**bits_1 - 1
    codes_1 = np.arange(max_code_1 + 1)
    xnorm_1 = codes_1 / max_code_1

    ideal_v_1 = vmin_1 + (vmax_1 - vmin_1) * xnorm_1
    bow_shape_1 = 4.0 * xnorm_1 * (1.0 - xnorm_1)

    rng_1 = np.random.default_rng(int(p1_seed.value))
    measured_v_1 = ideal_v_1 + bow_1 * bow_shape_1 + rng_1.normal(0.0, noise_1, size=len(codes_1))

    if fit_mode_1 == "End-point based":
        slope_1 = (measured_v_1[-1] - measured_v_1[0]) / max_code_1
        intercept_1 = measured_v_1[0]
        ref_label_1 = "End-point line"
    else:
        slope_1, intercept_1 = np.polyfit(codes_1, measured_v_1, 1)
        ref_label_1 = "Best-fit line"

    ref_v_1 = slope_1 * codes_1 + intercept_1
    lsb_1 = (vmax_1 - vmin_1) / max_code_1
    inl_lsb_1 = (measured_v_1 - ref_v_1) / lsb_1
    dnl_lsb_1 = np.diff(measured_v_1) / lsb_1 - 1.0

    p1_df = pl.DataFrame(
        {
            "Code": codes_1,
            "Ideal_V": ideal_v_1,
            "Measured_V": measured_v_1,
            "Reference_V": ref_v_1,
            "Error_vs_Reference_V": measured_v_1 - ref_v_1,
            "INL_LSB": inl_lsb_1,
        }
    )

    p1_dnl_df = pl.DataFrame(
        {
            "Transition_Code": codes_1[:-1],
            "DNL_LSB": dnl_lsb_1,
        }
    )

    p1_transfer_df = pl.DataFrame(
        {
            "Code": np.concatenate([codes_1, codes_1]),
            "Voltage": np.concatenate([measured_v_1, ref_v_1]),
            "Series": ["Measured"] * len(codes_1) + [ref_label_1] * len(codes_1),
        }
    )

    p1_inl_df = pl.DataFrame(
        {
            "Code": codes_1,
            "INL_LSB": inl_lsb_1,
        }
    )

    p1_dnl_df_plot = pl.DataFrame(
        {
            "Transition_Code": codes_1[:-1],
            "DNL_LSB": dnl_lsb_1,
        }
    )

    fig1 = px.line(
        p1_transfer_df,
        x="Code",
        y="Voltage",
        color="Series",
        title="DAC Transfer Curve and Reference Line",
        labels={"Code": "DAC Code", "Voltage": "Output Voltage (V)", "Series": ""},
    )
    fig1.update_traces(line=dict(width=2))
    fig1.update_traces(line_shape="hv", selector={"name": "Measured"})
    fig1.update_layout(legend_title_text="")

    fig1_inl = px.line(
        p1_inl_df,
        x="Code",
        y="INL_LSB",
        title="Integral Linearity Error (INL)",
        labels={"Code": "DAC Code", "INL_LSB": "INL (LSB)"},
    )
    fig1_inl.update_traces(line_shape="hv")
    fig1_inl.add_hline(y=0.0, line_width=1, line_dash="dash", line_color="#B0B0B0")

    inl_min_idx = int(np.argmin(inl_lsb_1))
    inl_max_idx = int(np.argmax(inl_lsb_1))
    inl_min_x, inl_min_y = int(codes_1[inl_min_idx]), float(inl_lsb_1[inl_min_idx])
    inl_max_x, inl_max_y = int(codes_1[inl_max_idx]), float(inl_lsb_1[inl_max_idx])
    fig1_inl.add_scatter(
        x=[inl_min_x, inl_max_x],
        y=[inl_min_y, inl_max_y],
        mode="markers+text",
        marker=dict(symbol="circle", size=10, color=["crimson", "seagreen"]),
        text=[f"Min: {inl_min_y:.3f}", f"Max: {inl_max_y:.3f}"],
        textposition="top center",
        name="INL Extrema",
    )

    fig1_dnl = px.line(
        p1_dnl_df_plot,
        x="Transition_Code",
        y="DNL_LSB",
        title="Differential Linearity Error (DNL)",
        labels={"Transition_Code": "Transition Code", "DNL_LSB": "DNL (LSB)"},
    )
    fig1_dnl.update_traces(line_shape="hv")
    fig1_dnl.add_hline(y=0.0, line_width=1, line_dash="dash", line_color="#B0B0B0")

    dnl_min_idx = int(np.argmin(dnl_lsb_1))
    dnl_max_idx = int(np.argmax(dnl_lsb_1))
    dnl_min_x, dnl_min_y = int(codes_1[:-1][dnl_min_idx]), float(dnl_lsb_1[dnl_min_idx])
    dnl_max_x, dnl_max_y = int(codes_1[:-1][dnl_max_idx]), float(dnl_lsb_1[dnl_max_idx])
    fig1_dnl.add_scatter(
        x=[dnl_min_x, dnl_max_x],
        y=[dnl_min_y, dnl_max_y],
        mode="markers+text",
        marker=dict(symbol="circle", size=10, color=["crimson", "seagreen"]),
        text=[f"Min: {dnl_min_y:.3f}", f"Max: {dnl_max_y:.3f}"],
        textposition="top center",
        name="DNL Extrema",
    )

    # Keep output manageable for very high resolutions.
    p1_show_df = p1_df if p1_df.height <= 4096 else p1_df.head(4096)
    p1_dnl_show_df = p1_dnl_df if p1_dnl_df.height <= 4096 else p1_dnl_df.head(4096)
    return fig1, fig1_dnl, fig1_inl, p1_show_df


@app.cell
def _(mo):
    # Page 2 controls
    p2_vmin = mo.ui.number(
        start=-20,
        stop=20,
        step=0.01,
        value=0.0,
        label="Min output voltage (V)",
    )
    p2_vmax = mo.ui.number(
        start=-20,
        stop=20,
        step=0.01,
        value=5.0,
        label="Max output voltage (V)",
    )
    p2_resolution = mo.ui.dropdown(
        options={"8-bit": 8, "10-bit": 10, "12-bit": 12, "14-bit": 14},
        value='8-bit',
        label="DAC resolution",
    )
    p2_periods = mo.ui.number(
        start=1,
        stop=100,
        step=1,
        value=5,
        label="# sine wave periods",
    )
    p2_samples = mo.ui.number(
        start=64,
        stop=131072,
        step=64,
        value=4096,
        label="# sample points",
    )
    p2_fs = mo.ui.number(
        start=1,
        stop=50_000_000,
        step=1,
        value=1_000_000,
        label="DAC conversion rate (samples/s)",
    )

    p2_controls = mo.vstack(
        [
            mo.md("## Slide 5: DAC Sine Wave Tests"),
            mo.hstack([p2_vmin, p2_vmax, p2_resolution]),
            mo.hstack([p2_periods, p2_samples, p2_fs]),
        ]
    )
    return (
        p2_controls,
        p2_fs,
        p2_periods,
        p2_resolution,
        p2_samples,
        p2_vmax,
        p2_vmin,
    )


@app.cell
def _(
    np,
    p2_fs,
    p2_periods,
    p2_resolution,
    p2_samples,
    p2_vmax,
    p2_vmin,
    pl,
    px,
):
    # Page 2 calculations and plots
    vmin_2 = float(p2_vmin.value)
    vmax_2 = float(p2_vmax.value)
    bits_2 = int(p2_resolution.value)
    periods_2 = int(p2_periods.value)
    n_2 = int(p2_samples.value)
    fs_2 = float(p2_fs.value)

    if vmax_2 <= vmin_2:
        vmax_2 = vmin_2 + 1e-6

    n_2 = max(64, n_2)
    periods_2 = max(1, min(periods_2, n_2 // 2))

    t_2 = np.arange(n_2) / fs_2
    f0_2 = periods_2 * fs_2 / n_2

    offset_2 = 0.5 * (vmax_2 + vmin_2)
    amp_2 = 0.49 * (vmax_2 - vmin_2)
    ideal_wave_2 = offset_2 + amp_2 * np.sin(2.0 * np.pi * f0_2 * t_2)

    max_code_2 = 2**bits_2 - 1
    codes_2 = np.clip(
        np.round((ideal_wave_2 - vmin_2) / (vmax_2 - vmin_2) * max_code_2),
        0,
        max_code_2,
    ).astype(int)
    quant_wave_2 = vmin_2 + (codes_2 / max_code_2) * (vmax_2 - vmin_2)

    ac_2 = quant_wave_2 - np.mean(quant_wave_2)
    fft_2 = np.fft.rfft(ac_2)
    freqs_2 = np.fft.rfftfreq(n_2, d=1.0 / fs_2)
    power_2 = np.abs(fft_2) ** 2

    # Coherent sampling: fundamental is at the programmed period bin.
    k_fund_2 = periods_2
    p_fund_2 = power_2[k_fund_2] if k_fund_2 < len(power_2) else 1e-30

    harmonic_bins_2 = []
    for h in range(2, 11):
        k_h = h * k_fund_2
        if k_h < len(power_2):
            harmonic_bins_2.append(k_h)

    p_harm_2 = float(np.sum(power_2[harmonic_bins_2])) if harmonic_bins_2 else 0.0

    excluded_2 = {0, k_fund_2, *harmonic_bins_2}
    noise_bins_2 = [k for k in range(1, len(power_2)) if k not in excluded_2]
    p_noise_2 = float(np.sum(power_2[noise_bins_2])) if noise_bins_2 else 1e-30

    snr_db_2 = 10.0 * np.log10(max(p_fund_2, 1e-30) / max(p_noise_2, 1e-30))
    thd_db_2 = 10.0 * np.log10(max(p_harm_2, 1e-30) / max(p_fund_2, 1e-30))

    spur_bins_2 = [k for k in range(1, len(power_2)) if k != k_fund_2]
    p_spur_2 = float(np.max(power_2[spur_bins_2])) if spur_bins_2 else 1e-30
    sfdr_db_2 = 10.0 * np.log10(max(p_fund_2, 1e-30) / max(p_spur_2, 1e-30))

    mag_db_2 = 10.0 * np.log10(np.maximum(power_2, 1e-30))
    mag_db_2 -= np.max(mag_db_2)

    # Plot the full capture span so the displayed periods match the selected value.
    max_plot_points_2 = 2000
    if n_2 > max_plot_points_2:
        plot_idx_2 = np.linspace(0, n_2 - 1, num=max_plot_points_2, dtype=int)
        plot_idx_2 = np.unique(plot_idx_2)
    else:
        plot_idx_2 = np.arange(n_2)

    # Add the endpoint at t = n/fs with the starting value to visualize full coherent cycles.
    t_plot_2 = np.append(t_2[plot_idx_2], n_2 / fs_2)
    y_plot_2 = np.append(quant_wave_2[plot_idx_2], quant_wave_2[0])

    fig2a = px.line(
        x=t_plot_2 * 1e6,
        y=y_plot_2,
        title="Quantized DAC Sine Wave (Full Capture, Time Domain)",
        labels={"x": "Time (us)", "y": "Voltage (V)"},
    )
    fig2a.update_traces(line=dict(width=1.5))

    fig2b = px.line(
        x=freqs_2 / 1e3,
        y=mag_db_2,
        title="Frequency Response (FFT Magnitude, dBFS)",
        labels={"x": "Frequency (kHz)", "y": "Magnitude (dBFS)"},
    )
    fig2b.update_traces(line=dict(width=1.5))
    fig2b.update_yaxes(range=[-180, 5])

    metrics_2 = pl.DataFrame(
        {
            "Metric": ["Fundamental Frequency (Hz)", "SNR (dB)", "THD (dB)", "SFDR (dB)"],
            "Value": [f0_2, snr_db_2, thd_db_2, sfdr_db_2],
        }
    )
    return fig2a, fig2b, metrics_2


@app.cell
def _(mo):
    endpoint_vs_best_fit = mo.vstack(
        [
            mo.md(
                """
                ## Slide 3: Endpoint vs Best-Fit Line Ideal Models
                When comparing measured DAC outputs to \"ideal\" values, two common reference models are used:

                ### **Endpoint Method**

                - Draw a straight line between the measured output at **code 0** and **code full-scale**.
                - Assumes the DAC should perfectly hit both endpoints.
                - Sensitive to noise or measurement error at the endpoints.
                - Often used in datasheets for simplicity.

                ### **Best-Fit Line Method**

                - Perform a **least-squares linear regression** on all measured output points.
                - Produces an \"ideal\" line that best matches the overall behavior.
                - Reduces sensitivity to endpoint noise.
                - Often used in precision characterization.

                Both methods produce slightly different INL and gain/offset error values.
                """
            )
        ]
    )

    # Slides 6 and 7 placeholders for presentation flow.
    page3 = mo.vstack(
        [
            mo.md("## Slide 6: DAC Dynamic Linearity (Placeholder)"),
            mo.md(
                "Add IMD, glitch impulse, settling behavior, or code-transition test content here."
            ),
        ]
    )

    page4 = mo.vstack(
        [
            mo.md("## Slide 7: ADC Validation Summary (Placeholder)"),
            mo.md(
                "Add ADC histogram tests, ENOB sweep, and final pass/fail summary charts here."
            ),
        ]
    )
    return endpoint_vs_best_fit, page3, page4


@app.cell
def _(fig1, mo, p1_controls, p1_show_df):
    page1 = mo.vstack(
        [
            p1_controls,
            mo.hstack(
                [
                    fig1,
                    mo.vstack(
                        [
                            mo.md("### Voltage for output codes"),
                            mo.ui.table(p1_show_df),
                        ]
                    ),
                ],
                widths=[3, 2],
            ),
            mo.md(
                "Displaying up to 4096 rows for performance. Increase resolution carefully for live demos."
            ),
        ]
    )

    page1
    return


@app.cell
def _(endpoint_vs_best_fit):
    endpoint_vs_best_fit
    return


@app.cell
def _(fig1_dnl, fig1_inl, mo):
    page1_linearity = mo.vstack(
        [
            mo.md("## Slide 4: DAC INL and DNL"),
            mo.hstack([fig1_inl, fig1_dnl]),
            # mo.md("### DNL by transition"),
            # mo.ui.table(p1_dnl_show_df),
        ]
    )

    page1_linearity
    return


@app.cell
def _(fig2a, fig2b, metrics_2, mo, p2_controls):
    page2 = mo.vstack(
        [
            p2_controls,
            fig2a,
            fig2b,
            mo.md("### Spectral Metrics"),
            mo.ui.table(metrics_2),
        ]
    )

    page2
    return


@app.cell
def _(page3):
    page3
    return


@app.cell
def _(page4):
    page4
    return


if __name__ == "__main__":
    app.run()
