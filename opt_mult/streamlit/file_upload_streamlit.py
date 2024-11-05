import streamlit as st

# Title of the web app
st.title('File Upload Streamlit App')

# File uploader widget
markscheme_file = st.file_uploader("Choose a markscheme file to upload", type=["pdf"])

answer_sheet_file = st.file_uploader("Choose a answer file to upload", type=["pdf"])

# Check if the user has uploaded a file
if markscheme_file is not None and answer_sheet_file is not None:
    # Display the filename after upload
    st.write(f'answersheet file: {answer_sheet_file.name}')
    st.write(f'markscheme file: {markscheme_file.name}')

    #TODO: load the images to numpy arrays, cycle through the images and return a mark

else:
    st.write('Please upload a file to continue.')