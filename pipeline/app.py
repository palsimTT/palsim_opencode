"""
PaLSim Gradio Web Interface(a test version for one player)
"""

import gradio as gr
import pandas as pd
import numpy as np

try:
    from .api import PaLSim
except ImportError:
    from api import PaLSim


# default player: M
api = PaLSim()

# Get valid strategy lists
valid_strategies = api.get_valid_strategies()
ST_OPTIONS = [""] + valid_strategies['st']
BP_OPTIONS = [""] + valid_strategies['bp']


def format_percentage(value, threshold=0.001):
    """Format as percentage, return empty string if below threshold"""
    if value < threshold:
        return ""
    return f"{value * 100:.2f}%"


def get_default_tables():
    """Generate default tables with N/A values for initial display"""
    st_labels = api.st_labels
    bp_labels = api.bp_labels

    bp_df = pd.DataFrame(
        [["N/A"] * len(bp_labels)],
        columns=bp_labels,
        index=["Probability"]
    )

    st_df = pd.DataFrame(
        [["N/A"] * len(st_labels)],
        columns=st_labels,
        index=["Probability"]
    )

    joint_data = [["N/A"] * len(bp_labels) for _ in range(len(st_labels))]
    joint_df = pd.DataFrame(
        joint_data,
        columns=bp_labels,
        index=st_labels
    )
    joint_df = joint_df.reset_index()
    joint_df = joint_df.rename(columns={'index': 'ST \\ BP'})
    
    winrate_df = pd.DataFrame(columns=['ST', 'BP', 'Probability', 'Win Rate'])
    
    expected_winrate = "N/A"
    
    return bp_df, st_df, joint_df, winrate_df, expected_winrate


def simulate(st_t2, bp_t2, st_t1, bp_t1):
    """
    Execute simulation prediction
    attention: model needs at least one stroke as a context, because serve is out of our consideration
    """
    st_t2 = st_t2 if st_t2 else None
    bp_t2 = bp_t2 if bp_t2 else None
    st_t1 = st_t1 if st_t1 else None
    bp_t1 = bp_t1 if bp_t1 else None
    
    result = api.predict(st_t2=st_t2, bp_t2=bp_t2, st_t1=st_t1, bp_t1=bp_t1)
    
    bp_labels = result['bp_labels']
    bp_probs = result['bp_probs']
    bp_df = pd.DataFrame(
        [[format_percentage(p) for p in bp_probs]],
        columns=bp_labels,
        index=["Probability"]
    )
    
    st_labels = result['st_labels']
    st_probs = result['st_probs']
    st_df = pd.DataFrame(
        [[format_percentage(p) for p in st_probs]],
        columns=st_labels,
        index=["Probability"]
    )
    
    joint = np.array(result['joint'])
    
    flat_indices = np.argsort(joint.flatten())[::-1][:5]
    top5_positions = set()
    for idx in flat_indices:
        i, j = divmod(idx, len(bp_labels))
        top5_positions.add((i, j))
    
    joint_data = []
    for i in range(len(st_labels)):
        row = []
        for j in range(len(bp_labels)):
            val = joint[i, j]
            if val < 0.001:
                row.append("")
            elif (i, j) in top5_positions:
                row.append(f"[{val * 100:.2f}%]")  # Bracket for top 5
            else:
                row.append(f"{val * 100:.2f}%")
        joint_data.append(row)
    
    joint_df = pd.DataFrame(
        joint_data,
        columns=bp_labels,
        index=st_labels
    )
    # Reset index to make ST labels visible as a column
    joint_df = joint_df.reset_index()
    joint_df = joint_df.rename(columns={'index': 'ST \\ BP'})
    
    winrate_matrix = np.array(result['winrate_matrix'])
    winrate_rows = []
    for i in range(len(st_labels)):
        for j in range(len(bp_labels)):
            prob = joint[i, j]
            if prob >= 0.001:  # Only show if probability > 0.1%
                winrate_rows.append({
                    'ST': st_labels[i],
                    'BP': bp_labels[j],
                    'Probability': f"{prob * 100:.2f}%",
                    'Win Rate': f"{winrate_matrix[i, j] * 100:.2f}%"
                })
    
    winrate_rows.sort(key=lambda x: float(x['Probability'].replace('%', '')), reverse=True)
    winrate_df = pd.DataFrame(winrate_rows)
    
    expected_winrate = f"{result['expected_winrate'] * 100:.2f}%"
    
    return bp_df, st_df, joint_df, winrate_df, expected_winrate


def create_interface():
    """Create Gradio interface"""
    
    with gr.Blocks(title="PaLSim_test version") as demo:
        gr.Markdown("# PaLSim_test version")
        gr.Markdown("Predict player's stroke technique (ST), ball placement (BP) distribution and win rate")
        
        gr.Markdown("## Context")
        gr.Markdown("Select historical stroke technique and ball placement (can be left empty)")
        
        with gr.Row():
            st_t2 = gr.Dropdown(
                choices=ST_OPTIONS,
                label="ST (t-2)",
                value="",
                info="Stroke technique at t-2"
            )
            bp_t2 = gr.Dropdown(
                choices=BP_OPTIONS,
                label="BP (t-2)",
                value="",
                info="Ball placement at t-2"
            )
            st_t1 = gr.Dropdown(
                choices=ST_OPTIONS,
                label="ST (t-1)",
                value="",
                info="Stroke technique at t-1"
            )
            bp_t1 = gr.Dropdown(
                choices=BP_OPTIONS,
                label="BP (t-1)",
                value="",
                info="Ball placement at t-1"
            )
        
        simulate_btn = gr.Button("Simulate", variant="primary", size="lg")
        
        gr.Markdown("---")
        gr.Markdown("## Results")
        
        gr.Markdown("### BP Probability")
        bp_table = gr.Dataframe(
            label="",
            interactive=False,
            wrap=True
        )
        
        gr.Markdown("### ST Probability")
        st_table = gr.Dataframe(
            label="",
            interactive=False,
            wrap=True
        )
        
        gr.Markdown("### Joint Distribution")
        gr.Markdown("*Row: Stroke Technique (ST), Column: Ball Placement (BP). Top 5 combinations are marked with [brackets].*")
        joint_table = gr.Dataframe(
            label="",
            interactive=False,
            wrap=True,
            elem_classes=["joint-table"]
        )
        
        gr.Markdown("### Win Rate by Combination")
        gr.Markdown("*Only showing combinations with probability > 0.1%*")
        winrate_table = gr.Dataframe(
            label="",
            interactive=False,
            wrap=True
        )
        
        gr.Markdown("### Expected Win Rate")
        expected_winrate = gr.Textbox(
            label="",
            interactive=False,
            elem_classes=["expected-winrate"]
        )
        
        simulate_btn.click(
            fn=simulate,
            inputs=[st_t2, bp_t2, st_t1, bp_t1],
            outputs=[bp_table, st_table, joint_table, winrate_table, expected_winrate]
        )
        
        demo.load(
            fn=get_default_tables,
            outputs=[bp_table, st_table, joint_table, winrate_table, expected_winrate]
        )
        
        gr.Markdown("---")
        gr.Markdown(f"*Device: {api.device}*")
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False
    )
