{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNUhCtM04bzTxCvUqz2GqBZ",
      "include_colab_link": True
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/solanki828/Jai-Solanki/blob/main/Capstone%20Project%20based%20on%20Image%20Classification.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "nk17Awvgwlkk",
        "outputId": "3b564134-19ad-4a26-8d92-8be3849fcea4"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting streamlit\n",
            "  Downloading streamlit-1.66.1-py3-none-any.whl.metadata (10 kB)\n",
            "Requirement already satisfied: tensorflow in /usr/local/lib/python3.13/dist-packages (2.21.0)\n",
            "Requirement already satisfied: numpy in /usr/local/lib/python3.13/dist-packages (2.4.6)\n",
            "Requirement already satisfied: pillow in /usr/local/lib/python3.13/dist-packages (11.3.0)\n",
            "Requirement already satisfied: altair!=5.4.0,!=5.4.1,<7,>=5.0.0 in /usr/local/lib/python3.13/dist-packages (from streamlit) (5.5.0)\n",
            "Requirement already satisfied: click<9,>=7.0 in /usr/local/lib/python3.13/dist-packages (from streamlit) (8.5.0)\n",
            "Requirement already satisfied: packaging>=20 in /usr/local/lib/python3.13/dist-packages (from streamlit) (26.3)\n",
            "Requirement already satisfied: pandas<4,>=1.5.3 in /usr/local/lib/python3.13/dist-packages (from streamlit) (2.2.3)\n",
            "Collecting pydeck<1,>=0.8.0b4 (from streamlit)\n",
            "  Downloading pydeck-0.9.3-py2.py3-none-any.whl.metadata (4.2 kB)\n",
            "Requirement already satisfied: protobuf<8,>=5.26.1 in /usr/local/lib/python3.13/dist-packages (from streamlit) (6.33.6)\n",
            "Requirement already satisfied: pyarrow!=25.0.0,<27,>=10.0.1 in /usr/local/lib/python3.13/dist-packages (from streamlit) (23.0.1)\n",
            "Requirement already satisfied: requests<3,>=2.27 in /usr/local/lib/python3.13/dist-packages (from streamlit) (2.32.4)\n",
            "Requirement already satisfied: typing-extensions<5,>=4.10.0 in /usr/local/lib/python3.13/dist-packages (from streamlit) (4.16.0)\n",
            "Requirement already satisfied: starlette<2,>=0.46.0 in /usr/local/lib/python3.13/dist-packages (from streamlit) (1.10.0)\n",
            "Requirement already satisfied: uvicorn<1,>=0.30.0 in /usr/local/lib/python3.13/dist-packages (from streamlit) (0.54.0)\n",
            "Requirement already satisfied: anyio<5,>=4.0.0 in /usr/local/lib/python3.13/dist-packages (from streamlit) (4.15.1)\n",
            "Requirement already satisfied: python-multipart<1,>=0.0.10 in /usr/local/lib/python3.13/dist-packages (from streamlit) (0.0.32)\n",
            "Requirement already satisfied: websockets<18,>=12.0.0 in /usr/local/lib/python3.13/dist-packages (from streamlit) (15.0.1)\n",
            "Requirement already satisfied: itsdangerous<3,>=2.1.2 in /usr/local/lib/python3.13/dist-packages (from streamlit) (2.2.0)\n",
            "Requirement already satisfied: watchdog<7,>=2.1.5 in /usr/local/lib/python3.13/dist-packages (from streamlit) (6.0.0)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.13/dist-packages (from altair!=5.4.0,!=5.4.1,<7,>=5.0.0->streamlit) (3.1.6)\n",
            "Requirement already satisfied: jsonschema>=3.0 in /usr/local/lib/python3.13/dist-packages (from altair!=5.4.0,!=5.4.1,<7,>=5.0.0->streamlit) (4.26.0)\n",
            "Requirement already satisfied: narwhals>=1.14.2 in /usr/local/lib/python3.13/dist-packages (from altair!=5.4.0,!=5.4.1,<7,>=5.0.0->streamlit) (2.28.0)\n",
            "Requirement already satisfied: idna>=2.8 in /usr/local/lib/python3.13/dist-packages (from anyio<5,>=4.0.0->streamlit) (3.20)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.13/dist-packages (from pandas<4,>=1.5.3->streamlit) (2.9.0.post0)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.13/dist-packages (from pandas<4,>=1.5.3->streamlit) (2025.2)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.13/dist-packages (from pandas<4,>=1.5.3->streamlit) (2026.6)\n",
            "Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.13/dist-packages (from requests<3,>=2.27->streamlit) (3.4.9)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.13/dist-packages (from requests<3,>=2.27->streamlit) (2.5.0)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.13/dist-packages (from requests<3,>=2.27->streamlit) (2026.10.23)\n",
            "Requirement already satisfied: h11>=0.8 in /usr/local/lib/python3.13/dist-packages (from uvicorn<1,>=0.30.0->streamlit) (0.16.0)\n",
            "Requirement already satisfied: absl-py>=1.0.0 in /usr/local/lib/python3.13/dist-packages (from tensorflow) (2.0.0)\n",
            "Requirement already satisfied: astunparse>=1.6.0 in /usr/local/lib/python3.13/dist-packages (from tensorflow) (1.6.3)\n",
            "Requirement already satisfied: flatbuffers>=25.9.23 in /usr/local/lib/python3.13/dist-packages (from tensorflow) (25.12.19)\n",
            "Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in /usr/local/lib/python3.13/dist-packages (from tensorflow) (0.7.0)\n",
            "Requirement already satisfied: google_pasta>=0.1.1 in /usr/local/lib/python3.13/dist-packages (from tensorflow) (0.2.0)\n",
            "Requirement already satisfied: libclang>=13.0.0 in /usr/local/lib/python3.13/dist-packages (from tensorflow) (18.1.1)\n",
            "Requirement already satisfied: opt_einsum>=2.3.2 in /usr/local/lib/python3.13/dist-packages (from tensorflow) (3.4.0)\n",
            "Requirement already satisfied: setuptools in /usr/local/lib/python3.13/dist-packages (from tensorflow) (80.10.2)\n",
            "Requirement already satisfied: six>=1.12.0 in /usr/local/lib/python3.13/dist-packages (from tensorflow) (1.17.0)\n",
            "Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.13/dist-packages (from tensorflow) (3.3.0)\n",
            "Requirement already satisfied: wrapt>=1.11.0 in /usr/local/lib/python3.13/dist-packages (from tensorflow) (2.5.1)\n",
            "Requirement already satisfied: grpcio<2.0,>=1.24.3 in /usr/local/lib/python3.13/dist-packages (from tensorflow) (1.85.0)\n",
            "Requirement already satisfied: keras>=3.12.0 in /usr/local/lib/python3.13/dist-packages (from tensorflow) (3.13.2)\n",
            "Requirement already satisfied: h5py<3.15.0,>=3.11.0 in /usr/local/lib/python3.13/dist-packages (from tensorflow) (3.14.0)\n",
            "Requirement already satisfied: ml_dtypes<1.0.0,>=0.5.1 in /usr/local/lib/python3.13/dist-packages (from tensorflow) (0.6.0)\n",
            "Requirement already satisfied: wheel<1.0,>=0.23.0 in /usr/local/lib/python3.13/dist-packages (from astunparse>=1.6.0->tensorflow) (0.48.0)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.13/dist-packages (from jinja2->altair!=5.4.0,!=5.4.1,<7,>=5.0.0->streamlit) (3.0.4)\n",
            "Requirement already satisfied: attrs>=22.2.0 in /usr/local/lib/python3.13/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=5.0.0->streamlit) (26.1.0)\n",
            "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.13/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=5.0.0->streamlit) (2025.9.1)\n",
            "Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.13/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=5.0.0->streamlit) (0.37.0)\n",
            "Requirement already satisfied: rpds-py>=0.25.0 in /usr/local/lib/python3.13/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=5.0.0->streamlit) (2026.9.1)\n",
            "Requirement already satisfied: rich in /usr/local/lib/python3.13/dist-packages (from keras>=3.12.0->tensorflow) (13.9.4)\n",
            "Requirement already satisfied: namex in /usr/local/lib/python3.13/dist-packages (from keras>=3.12.0->tensorflow) (0.1.0)\n",
            "Requirement already satisfied: optree in /usr/local/lib/python3.13/dist-packages (from keras>=3.12.0->tensorflow) (0.20.0)\n",
            "Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.13/dist-packages (from rich->keras>=3.12.0->tensorflow) (4.2.0)\n",
            "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.13/dist-packages (from rich->keras>=3.12.0->tensorflow) (2.21.0)\n",
            "Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.13/dist-packages (from markdown-it-py>=2.2.0->rich->keras>=3.12.0->tensorflow) (0.1.2)\n",
            "Downloading streamlit-1.66.1-py3-none-any.whl (10.4 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m10.4/10.4 MB\u001b[0m \u001b[31m80.7 MB/s\u001b[0m  \u001b[33m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading pydeck-0.9.3-py2.py3-none-any.whl (11.4 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m11.4/11.4 MB\u001b[0m \u001b[31m106.6 MB/s\u001b[0m  \u001b[33m0:00:00\u001b[0m\n",
            "\u001b[?25hInstalling collected packages: pydeck, streamlit\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m2/2\u001b[0m [streamlit]\n",
            "\u001b[1A\u001b[2KSuccessfully installed pydeck-0.9.3 streamlit-1.66.1\n"
          ]
        }
      ],
      "source": [
        "pip install streamlit tensorflow numpy pillow"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Streamlight"
      ],
      "metadata": {
        "id": "Qj4QU3BEwxck"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "import streamlit as st\n",
        "import tensorflow as tf\n",
        "import numpy as np\n",
        "\n",
        "# Page configuration\n",
        "st.set_page_config(\n",
        "    page_title=\"Flower Classifier\",\n",
        "    page_icon=\"🌸\",\n",
        "    layout=\"centered\"\n",
        ")\n",
        "\n",
        "st.title(\"🌸 AI Flower Classification\")\n",
        "st.write(\"Upload a flower image to predict its class.\")\n",
        "\n",
        "# Load the trained model\n",
        "@st.cache_resource\n",
        "def load_model():\n",
        "    return tf.keras.models.load_model(\"flower_classifier.keras\")\n",
        "\n",
        "try:\n",
        "    model = load_model()\n",
        "except Exception as e:\n",
        "    st.error(f\"Could not load model: {e}\")\n",
        "    st.stop()\n",
        "\n",
        "# Use the class_names list from your dataset\n",
        "# Example:\n",
        "# class_names = [\"daisy\", \"dandelion\", \"roses\", \"sunflowers\", \"tulips\"]\n",
        "\n",
        "uploaded_file = st.file_uploader(\n",
        "    \"Choose a flower image\",\n",
        "    type=[\"jpg\", \"jpeg\", \"png\"]\n",
        ")\n",
        "\n",
        "if uploaded_file is not None:\n",
        "\n",
        "    # Display uploaded image\n",
        "    st.image(\n",
        "        uploaded_file,\n",
        "        caption=\"Uploaded Flower Image\",\n",
        "        width=300\n",
        "    )\n",
        "\n",
        "    if st.button(\"Predict Flower\"):\n",
        "\n",
        "        with st.spinner(\"Analyzing flower image...\"):\n",
        "\n",
        "            # Load and preprocess image\n",
        "            img = tf.keras.utils.load_img(\n",
        "                uploaded_file,\n",
        "                target_size=(128, 128)\n",
        "            )\n",
        "\n",
        "            x = tf.keras.utils.img_to_array(img)\n",
        "            x = x / 255.0\n",
        "            x = tf.expand_dims(x, 0)\n",
        "\n",
        "            # Run prediction\n",
        "            predictions = model.predict(x, verbose=0)\n",
        "            probabilities = predictions[0]\n",
        "\n",
        "            predicted_index = np.argmax(probabilities)\n",
        "            confidence = float(probabilities[predicted_index])\n",
        "\n",
        "            # Display prediction\n",
        "            st.success(\n",
        "                f\"Predicted Flower: {class_names[predicted_index]}\"\n",
        "            )\n",
        "\n",
        "            st.metric(\n",
        "                \"Confidence\",\n",
        "                f\"{confidence * 100:.2f}%\"\n",
        "            )\n",
        "\n",
        "            # Display probabilities for all classes\n",
        "            st.subheader(\"Prediction Probabilities\")\n",
        "\n",
        "            st.bar_chart({\n",
        "                class_names[i]: float(probabilities[i])\n",
        "                for i in range(len(class_names))\n",
        "            })\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xwt_vco4ww99",
        "outputId": "d0df1ed1-1f03-42ab-ca44-aceffd48c4b8"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2026-11-03 14:15:15.785 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-11-03 14:15:15.786 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-11-03 14:15:16.088 \n",
            "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
            "  command:\n",
            "\n",
            "    streamlit run /usr/local/lib/python3.13/dist-packages/colab_kernel_launcher.py [ARGUMENTS]\n",
            "2026-11-03 14:15:16.089 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-11-03 14:15:16.090 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-11-03 14:15:16.091 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-11-03 14:15:16.093 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-11-03 14:15:16.095 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-11-03 14:15:16.098 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-11-03 14:15:16.101 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-11-03 14:15:16.102 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-11-03 14:15:16.103 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-11-03 14:15:16.104 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-11-03 14:15:16.105 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-11-03 14:15:16.106 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-11-03 14:15:16.108 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-11-03 14:15:16.109 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-11-03 14:15:16.110 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
          ]
        }
      ]
    }
  ]
}
