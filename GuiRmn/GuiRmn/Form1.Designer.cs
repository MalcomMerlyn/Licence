namespace GuiRmn
{
    partial class Form1
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(Form1));
            this.button_path = new System.Windows.Forms.Button();
            this.textbox_path = new System.Windows.Forms.TextBox();
            this.text_h = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.text_w = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.Start = new System.Windows.Forms.Button();
            this.radioButton1 = new System.Windows.Forms.RadioButton();
            this.label4 = new System.Windows.Forms.Label();
            this.SuspendLayout();
            // 
            // button_path
            // 
            this.button_path.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("button_path.BackgroundImage")));
            this.button_path.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Stretch;
            this.button_path.Location = new System.Drawing.Point(1018, 25);
            this.button_path.Name = "button_path";
            this.button_path.Size = new System.Drawing.Size(61, 58);
            this.button_path.TabIndex = 0;
            this.button_path.UseVisualStyleBackColor = true;
            this.button_path.Click += new System.EventHandler(this.button_path_Click);
            // 
            // textbox_path
            // 
            this.textbox_path.BackColor = System.Drawing.SystemColors.MenuHighlight;
            this.textbox_path.Font = new System.Drawing.Font("Times New Roman", 20.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textbox_path.Location = new System.Drawing.Point(93, 32);
            this.textbox_path.Name = "textbox_path";
            this.textbox_path.Size = new System.Drawing.Size(919, 39);
            this.textbox_path.TabIndex = 1;
            this.textbox_path.Text = "C:\\Users\\Batman\\Desktop\\Computer Science\\Licence\\Rmn Render\\Data";
            this.textbox_path.TextChanged += new System.EventHandler(this.textBox1_TextChanged);
            // 
            // text_h
            // 
            this.text_h.BackColor = System.Drawing.SystemColors.MenuHighlight;
            this.text_h.Font = new System.Drawing.Font("Microsoft Sans Serif", 24F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.text_h.Location = new System.Drawing.Point(200, 120);
            this.text_h.Name = "text_h";
            this.text_h.Size = new System.Drawing.Size(150, 44);
            this.text_h.TabIndex = 2;
            this.text_h.Text = "512";
            this.text_h.TextChanged += new System.EventHandler(this.text_h_TextChanged);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Font = new System.Drawing.Font("Microsoft Sans Serif", 24F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label1.Location = new System.Drawing.Point(12, 126);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(167, 37);
            this.label1.TabIndex = 3;
            this.label1.Text = "Resolution";
            this.label1.Click += new System.EventHandler(this.label1_Click);
            // 
            // text_w
            // 
            this.text_w.BackColor = System.Drawing.SystemColors.MenuHighlight;
            this.text_w.Font = new System.Drawing.Font("Microsoft Sans Serif", 24F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.text_w.Location = new System.Drawing.Point(402, 120);
            this.text_w.Name = "text_w";
            this.text_w.Size = new System.Drawing.Size(150, 44);
            this.text_w.TabIndex = 4;
            this.text_w.Text = "512";
            this.text_w.TextChanged += new System.EventHandler(this.textBox1_TextChanged_1);
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Font = new System.Drawing.Font("Microsoft Sans Serif", 26.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label2.Location = new System.Drawing.Point(356, 123);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(40, 39);
            this.label2.TabIndex = 5;
            this.label2.Text = "X";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Font = new System.Drawing.Font("Microsoft Sans Serif", 21.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label3.Location = new System.Drawing.Point(13, 32);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(74, 33);
            this.label3.TabIndex = 6;
            this.label3.Text = "Path";
            this.label3.Click += new System.EventHandler(this.label3_Click);
            // 
            // Start
            // 
            this.Start.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("Start.BackgroundImage")));
            this.Start.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Stretch;
            this.Start.Font = new System.Drawing.Font("Microsoft Sans Serif", 36F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.Start.Location = new System.Drawing.Point(746, 161);
            this.Start.Name = "Start";
            this.Start.Size = new System.Drawing.Size(220, 220);
            this.Start.TabIndex = 7;
            this.Start.UseVisualStyleBackColor = true;
            this.Start.Click += new System.EventHandler(this.Start_Click);
            // 
            // radioButton1
            // 
            this.radioButton1.AutoSize = true;
            this.radioButton1.BackColor = System.Drawing.Color.Goldenrod;
            this.radioButton1.Font = new System.Drawing.Font("Microsoft Sans Serif", 27.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.radioButton1.Location = new System.Drawing.Point(93, 213);
            this.radioButton1.Name = "radioButton1";
            this.radioButton1.Size = new System.Drawing.Size(338, 46);
            this.radioButton1.TabIndex = 8;
            this.radioButton1.TabStop = true;
            this.radioButton1.Text = "Internal Structure";
            this.radioButton1.UseVisualStyleBackColor = false;
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Font = new System.Drawing.Font("Microsoft Sans Serif", 24F, ((System.Drawing.FontStyle)((System.Drawing.FontStyle.Bold | System.Drawing.FontStyle.Italic))), System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label4.Location = new System.Drawing.Point(27, 445);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(772, 37);
            this.label4.TabIndex = 9;
            this.label4.Text = "Use W, A, S, D key to navigate arround the object";
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.Color.Goldenrod;
            this.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Stretch;
            this.ClientSize = new System.Drawing.Size(1091, 526);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.radioButton1);
            this.Controls.Add(this.Start);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.text_w);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.text_h);
            this.Controls.Add(this.textbox_path);
            this.Controls.Add(this.button_path);
            this.ForeColor = System.Drawing.SystemColors.ControlText;
            this.Name = "Form1";
            this.Text = "Form1";
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button button_path;
        private System.Windows.Forms.TextBox textbox_path;
        private System.Windows.Forms.TextBox text_h;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox text_w;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Button Start;
        private System.Windows.Forms.RadioButton radioButton1;
        private System.Windows.Forms.Label label4;
    }
}

