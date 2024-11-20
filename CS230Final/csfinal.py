""" 
Name: Fynn Schwichtenberg
CS230: Section 6 
Data: 
Which data set you used URL:
https://insideairbnb.com/boston/
https://ourairports.com/data/

Link to your web application on Streamlit Cloud (if posted)

Description: This program will help people determine how convinent it is to go to and from the airport to their airbnb. 
People often choose airbnbs based on how conventient it is to get to and from.
In this project I will use the Boston Airbnb Data (https://ourairports.com/data/) and the New England Airport data from (https://insideairbnb.com/) 
to generate a map using Stream lits map package and then Marker Cluster to cluster air bnb’s with similar proximities to selected airports. 
Users will be able to select an airport in the new england area and then see what Airbnbs have the best convenience score calculated based on how 
far away they are from the airport. Depending on the range they fall in, 
a different color will be attributed to the dot representing the one or many Air bnbs and how close they are.
 """


import pandas as pd
import streamlit as st
import pydeck as pdk
from haversine import haversine, Unit
import matplotlib.pyplot as plt

# Initial data structures
states = {
    "US-MA": "Massachusetts",
    "US-CT": "Connecticut",
    "US-RI": "Rhode Island",
    "US-NH": "New Hampshire",
    "US-VT": "Vermont",
    "US-ME": "Maine",
}

# Define Boston's coordinates
BOSTON_COORDS = (42.3601, -71.0589)

# Load New England airports data
def read_ne_airport():
    df = pd.read_csv("data/new_england_airports.csv").set_index("id")
    df = df.loc[
        df["iso_region"].isin(["US-MA", "US-CT", "US-RI", "US-NH", "US-VT", "US-ME"])
    ]
    df = df.loc[df["type"].isin(["small_airport", "medium_airport", "large_airport"])]
    df["iso_region"] = df["iso_region"].map(states)

    # Calculate distances from Boston and select the 10 closest airports
    df["distance_to_boston"] = df.apply(
        lambda row: haversine(BOSTON_COORDS, (row["latitude_deg"], row["longitude_deg"])),
        axis=1,
    )
    return df.nsmallest(10, "distance_to_boston")

# Load Airbnb data
def read_airbnb(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError as e:
        st.error(f"Airbnb data file not found: {e}")
        return pd.DataFrame()

# Calculate distance between two points on a map
def calculate_distance(loc1, loc2):
    return haversine(loc1, loc2, unit=Unit.MILES)

# Dynamic scoring based on min and max distances
def calculate_dynamic_score(distance, min_distance, max_distance):
    if max_distance == min_distance:  # Avoid division by zero
        return 10
    normalized_score = 10 - ((distance - min_distance) / (max_distance - min_distance)) * 9
    return round(max(1, min(normalized_score, 10)), 2)

# Calculate map bounds dynamically
def calculate_map_bounds(data, airport_coords):
    min_lat = min(data["latitude"].min(), airport_coords[0])
    max_lat = max(data["latitude"].max(), airport_coords[0])
    min_lon = min(data["longitude"].min(), airport_coords[1])
    max_lon = max(data["longitude"].max(), airport_coords[1])

    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2

    # Estimate zoom level based on the range of latitudes and longitudes
    lat_range = max_lat - min_lat
    lon_range = max_lon - min_lon
    max_range = max(lat_range, lon_range)
    zoom = 11 - max_range * 5  # Adjust factor for optimal zoom
    zoom = max(1, min(zoom, 15))  # Clamp zoom level between 1 and 15

    return center_lat, center_lon, zoom

# Render map with Pydeck
def pretty_map(airbnb_data, airport_coords, airport_name):
    st.subheader(f"Airbnb Convenience Map for {airport_name}")
    
    # Calculate distance to the airport
    airbnb_data.loc[:, "distance_to_airport"] = airbnb_data.apply(
        lambda row: calculate_distance(
            (row["latitude"], row["longitude"]), airport_coords
        ),
        axis=1,
    )

    # Calculate scores
    min_distance = airbnb_data["distance_to_airport"].min()
    max_distance = airbnb_data["distance_to_airport"].max()
    airbnb_data.loc[:, "score"] = airbnb_data["distance_to_airport"].apply(
        lambda distance: calculate_dynamic_score(distance, min_distance, max_distance)
    )

    # Dynamic map bounds
    center_lat, center_lon, zoom = calculate_map_bounds(airbnb_data, airport_coords)

    # Pydeck layers
    airport_layer = pdk.Layer(
        "ScatterplotLayer",
        data=pd.DataFrame(
            {"latitude": [airport_coords[0]], "longitude": [airport_coords[1]]}
        ),
        get_position=["longitude", "latitude"],
        get_radius=700,
        get_color=[255, 0, 0],
        pickable=True,
    )
    airbnb_layer = pdk.Layer(
        "ScatterplotLayer",
        data=airbnb_data,
        get_position=["longitude", "latitude"],
        get_radius=100,
        get_color="[score * 25, 255 - (score * 25), 150]",
        pickable=True,
    )

    tool_tip = {
        "html": """
            <b>Title</b> {name}<br>
            <b>Type:</b> {room_type}<br>
            <b>Price:</b> ${price}<br>
            <b>Convenience Score:</b> {score}
        """,
        "style": {"backgroundColor": "steelblue", "color": "white"},
    }

    deck = pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=zoom, pitch=0),
        layers=[airport_layer, airbnb_layer],
        tooltip=tool_tip,
    )

    st.pydeck_chart(deck)

# Sidebar with filters
def sidebar_filters(df):
    st.sidebar.title("Filters")

    # Multiselect for listing types
    types = df["room_type"].unique()
    selected_types = st.sidebar.multiselect(
        "Filter by Listing Type:",
        options=types,
        default=types,
    )

    # Check if the 'price' column exists
    if "price" not in df.columns:
        st.error("The 'price' column is missing in the data.")
        return selected_types, (0, 0)

    # Price slider
    min_price = int(df["price"].min())
    max_price = 2500 # Fixed the value to 2500 to make the output look better = for all entries = int(df["price"].max())
    price_range = st.sidebar.slider(
        "Filter by Price:",
        min_value=min_price,
        max_value=max_price,
        value=(min_price, max_price),
    )

    return selected_types, price_range

def plot_neighborhood_pie_chart(df, title="Airbnb Neighborhood Distribution"):
  
    # Ensure the column exists
    if "neighbourhood" not in df.columns:
        st.warning("The 'neighbourhood' column is missing in the dataset. Unable to plot pie chart.")
        return

    # Group by neighborhood and count occurrences
    neighborhood_counts = df["neighbourhood"].value_counts()

    # Calculate percentages
    total = neighborhood_counts.sum()
    percentages = (neighborhood_counts / total) * 100

    # Group neighborhoods with <5% into "Other"
    grouped_counts = neighborhood_counts[percentages >= 5].copy()
    grouped_counts["Other"] = neighborhood_counts[percentages < 5].sum()

    # Ensure there's at least one valid category to plot
    if grouped_counts.sum() == 0:
        st.warning("No data available.")
        return
    
    custom_colors = ['#7c1158', '#4421af', '#5ad45a', '#FFCC99', '#FFD700', '#b30000','#5bb45b']

    # Create pie chart
    fig, ax = plt.subplots(figsize=(7, 7))  # Slightly larger figure for better visibility
    wedges, texts, autotexts = ax.pie(
        grouped_counts,
        labels=grouped_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=custom_colors[:len(grouped_counts)],  # Limit colors to the number of wedges

        textprops={'fontsize': 10},  # Consistent font size for labels
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'}  # Add edge to chart
    )

    # Adds a styled title to the chart
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    st.pyplot(fig)

   
# Main app
def main():
    st.title("Airport-Airbnb Convenience Map")
    st.markdown("Find Airbnbs with the best convenience scores near selected New England airports.")

    # Load datasets
    airports = read_ne_airport()
    airbnb_data = read_airbnb("data/boston-airbnb/listings.csv")

    # Sidebar filters
    selected_types, price_range = sidebar_filters(airbnb_data)

    # Filter data and explicitly create a copy
    filtered_data = airbnb_data[
        (airbnb_data["room_type"].isin(selected_types)) &
        (airbnb_data["price"] >= price_range[0]) &
        (airbnb_data["price"] <= price_range[1])
    ].copy()

    # Select airport
    selected_airport = st.selectbox(
        "Select an airport:", airports.index, format_func=lambda x: airports.loc[x]["name"]
    )
    airport_info = airports.loc[selected_airport]
    airport_coords = (airport_info["latitude_deg"], airport_info["longitude_deg"])
    airport_name = airport_info["name"]

    # Render map
    pretty_map(filtered_data, airport_coords, airport_name)

    
    plot_neighborhood_pie_chart(filtered_data, title="Neighborhood Distribution")
 
if __name__ == "__main__":
    main()
