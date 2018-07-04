using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System;
using System.Diagnostics;
using System.ComponentModel;

namespace GuiRmn
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void button_path_Click(object sender, EventArgs e)
        {
            string folderPath = textbox_path.Text;

            FolderBrowserDialog folderBrowserDialog = new FolderBrowserDialog();
            folderBrowserDialog.SelectedPath = folderPath;

            if (folderBrowserDialog.ShowDialog() == DialogResult.OK)
            {
                folderPath = folderBrowserDialog.SelectedPath;
            }

            textbox_path.Text = folderPath;
        }

        private void textBox1_TextChanged(object sender, EventArgs e)
        {

        }

        private void text_h_TextChanged(object sender, EventArgs e)
        {

        }

        private void label1_Click(object sender, EventArgs e)
        {

        }

        private void label3_Click(object sender, EventArgs e)
        {

        }

        private void textBox1_TextChanged_1(object sender, EventArgs e)
        {

        }

        private void Start_Click(object sender, EventArgs e)
        {
            string[] tokens = textbox_path.Text.Split('\\');
            string fileName = tokens[tokens.Length - 1];

            Process.Start(@"C:\Users\Batman\Desktop\Computer Science\Licence\Rmn Render\x64\Release\Rmn Render.exe",
                "\"" + textbox_path.Text + "\" " + fileName + " " + text_h.Text + " " + text_w.Text);
        }
    }
}
