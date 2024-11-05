import streamlit as st

# Title of the web app
st.title('Simple Streamlit App')

# User input for name
name = st.text_input('Enter your name:')

# Dropdown to select favorite color
color = st.selectbox('Choose your favorite color:', ['Red', 'Green', 'Blue', 'Yellow', 'Purple'])

# Display a personalized message and set background color
if name:
    st.write(f'Hello, {name}!')
    st.write(f'Your favorite color is {color}.')
    
    # Set background color based on user selection
    if color == 'Red':
        st.markdown(
            f'<style>body {{background-color: red;}}</style>',
            unsafe_allow_html=True
        )
    elif color == 'Green':
        st.markdown(
            f'<style>body {{background-color: green;}}</style>',
            unsafe_allow_html=True
        )
    elif color == 'Blue':
        st.markdown(
            f'<style>body {{background-color: blue;}}</style>',
            unsafe_allow_html=True
        )
    elif color == 'Yellow':
        st.markdown(
            f'<style>body {{background-color: yellow;}}</style>',
            unsafe_allow_html=True
        )
    elif color == 'Purple':
        st.markdown(
            f'<style>body {{background-color: purple;}}</style>',
            unsafe_allow_html=True
        )
else:
    st.write('Please enter your name to get started.')
