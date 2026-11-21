# -*- coding: utf-8 -*-
"""Build the 10-page course report for the MATLAB transformer project.

Numbers that depend on the experiments live in report/numbers.json so that the
prose never has to be edited by hand; collect_numbers.py writes that file from
the run logs and benchmark output.
"""
import json
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
REPORT = Path(__file__).resolve().parent
FIG = ROOT / "guppy_matlab" / "figures"
OUT = REPORT / "out"
OUT.mkdir(parents=True, exist_ok=True)

NUM = json.loads((REPORT / "numbers.json").read_text(encoding="utf-8"))

SONG, HEI, MONO, SERIF = "宋体", "黑体", "Consolas", "Times New Roman"


# --------------------------------------------------------------------------
# low level helpers
# --------------------------------------------------------------------------
def set_font(run, ascii_font=SERIF, ea_font=SONG, size=10.5, bold=False, color=None):
    run.font.name = ascii_font
    run.font.size = Pt(size)
    run.bold = bold
    if color is not None:
        run.font.color.rgb = RGBColor(*color)
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:ascii"), ascii_font)
    rFonts.set(qn("w:hAnsi"), ascii_font)
    rFonts.set(qn("w:eastAsia"), ea_font)


def shade(el, hex_fill):
    pPr = el.get_or_add_pPr() if hasattr(el, "get_or_add_pPr") else el
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    pPr.append(shd)


def para(doc, text="", size=10.5, bold=False, align=None, space_before=0,
         space_after=3, line=1.32, indent=0.0, ascii_font=SERIF, ea_font=SONG,
         color=None, keep_with_next=False):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.line_spacing = line
    if indent:
        pf.first_line_indent = Cm(indent)
    if align is not None:
        p.alignment = align
    if keep_with_next:
        pf.keep_with_next = True
    if text:
        set_font(p.add_run(text), ascii_font, ea_font, size, bold, color)
    return p


def rich(doc, parts, size=10.5, line=1.32, space_after=3, indent=0.0, align=None):
    """parts: list of (text, bold) or (text, bold, color)."""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(space_after)
    pf.line_spacing = line
    if indent:
        pf.first_line_indent = Cm(indent)
    if align is not None:
        p.alignment = align
    for item in parts:
        txt, bold = item[0], item[1]
        color = item[2] if len(item) > 2 else None
        set_font(p.add_run(txt), SERIF, SONG, size, bold, color)
    return p


def h1(doc, text, space_before=11):
    p = para(doc, text, size=13, bold=True, ascii_font=HEI, ea_font=HEI,
             space_before=space_before, space_after=5, line=1.1, keep_with_next=True)
    pPr = p._element.get_or_add_pPr()
    bdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "8")
    bottom.set(qn("w:space"), "2")
    bottom.set(qn("w:color"), "9AA5B1")
    bdr.append(bottom)
    pPr.append(bdr)
    return p


def h2(doc, text, space_before=7):
    return para(doc, text, size=11, bold=True, ascii_font=HEI, ea_font=HEI,
                space_before=space_before, space_after=3, line=1.15, keep_with_next=True)


def bullet(doc, parts, size=10.5, space_after=1.5):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.left_indent = Cm(0.55)
    pf.first_line_indent = Cm(-0.55)
    pf.space_before = Pt(0)
    pf.space_after = Pt(space_after)
    pf.line_spacing = 1.28
    set_font(p.add_run("· "), SERIF, SONG, size, True)
    for item in parts:
        txt, bold = item[0], item[1]
        set_font(p.add_run(txt), SERIF, SONG, size, bold)
    return p


def code_block(doc, text, caption=None):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    no_split(tbl.rows[0])
    cell = tbl.cell(0, 0)
    cell.width = Cm(16.2)
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), "F4F6F8")
    tcPr.append(shd)
    borders = OxmlElement("w:tcBorders")
    for side in ("top", "left", "bottom", "right"):
        e = OxmlElement(f"w:{side}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), "4")
        e.set(qn("w:color"), "D8DEE6")
        borders.append(e)
    tcPr.append(borders)

    cell.paragraphs[0].text = ""
    first = True
    for line in text.rstrip("\n").split("\n"):
        p = cell.paragraphs[0] if first else cell.add_paragraph()
        first = False
        pf = p.paragraph_format
        pf.space_before = Pt(0)
        pf.space_after = Pt(0)
        pf.line_spacing = 1.02
        r = p.add_run(line if line else " ")
        set_font(r, MONO, SONG, 8.2, False, (0x1F, 0x29, 0x33))
    if caption:
        para(doc, caption, size=9, align=WD_ALIGN_PARAGRAPH.CENTER,
             space_before=2, space_after=6, color=(0x55, 0x5F, 0x6B))
    else:
        para(doc, "", size=4, space_after=0, line=0.5)


def figure(doc, name, width_cm, caption):
    path = FIG / name
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True   # never orphan a caption from its figure
    if path.exists():
        p.add_run().add_picture(str(path), width=Cm(width_cm))
    else:
        set_font(p.add_run(f"[missing figure: {name}]"), SERIF, SONG, 10, False, (0xC0, 0x30, 0x30))
    para(doc, caption, size=9, align=WD_ALIGN_PARAGRAPH.CENTER,
         space_before=1, space_after=7, color=(0x33, 0x3B, 0x45))


def no_split(row, header=False):
    trPr = row._tr.get_or_add_trPr()
    e = OxmlElement("w:cantSplit")
    trPr.append(e)
    if header:
        h = OxmlElement("w:tblHeader")
        trPr.append(h)


def table(doc, header, rows, widths, caption=None, size=9, align_center=True, left_cols=None,
          keep_together=False):
    if caption:
        cp = para(doc, caption, size=9, align=WD_ALIGN_PARAGRAPH.CENTER,
                  space_before=5, space_after=2, color=(0x33, 0x3B, 0x45))
        cp.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1, cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    hdr = t.rows[0]
    for i, h in enumerate(header):
        c = hdr.cells[i]
        c.width = Cm(widths[i])
        c.paragraphs[0].text = ""
        pf = c.paragraphs[0].paragraph_format
        pf.space_before = Pt(1.5)
        pf.space_after = Pt(1.5)
        if align_center:
            c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = c.paragraphs[0].add_run(h)
        set_font(r, SERIF, HEI, size, True)
        # header bottom rule
        tcPr = c._tc.get_or_add_tcPr()
        borders = OxmlElement("w:tcBorders")
        b = OxmlElement("w:bottom")
        b.set(qn("w:val"), "single")
        b.set(qn("w:sz"), "8")
        b.set(qn("w:color"), "333B45")
        borders.append(b)
        top = OxmlElement("w:top")
        top.set(qn("w:val"), "single")
        top.set(qn("w:sz"), "12")
        top.set(qn("w:color"), "333B45")
        borders.insert(0, top)
        tcPr.append(borders)
    no_split(hdr, header=True)
    for row in rows:
        new_row = t.add_row()
        no_split(new_row)
        cells = new_row.cells
        for i, v in enumerate(row):
            c = cells[i]
            c.width = Cm(widths[i])
            c.paragraphs[0].text = ""
            pf = c.paragraphs[0].paragraph_format
            pf.space_before = Pt(1.2)
            pf.space_after = Pt(1.2)
            if align_center and i > 0 and (left_cols is None or i not in left_cols):
                c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = c.paragraphs[0].add_run(str(v))
            set_font(r, SERIF, SONG, size, False)
    # bottom rule of the whole table
    last = t.rows[-1]
    for c in last.cells:
        tcPr = c._tc.get_or_add_tcPr()
        borders = OxmlElement("w:tcBorders")
        b = OxmlElement("w:bottom")
        b.set(qn("w:val"), "single")
        b.set(qn("w:sz"), "12")
        b.set(qn("w:color"), "333B45")
        borders.append(b)
        tcPr.append(borders)
    if keep_together:
        for row in t.rows[:-1]:
            for c in row.cells:
                for pp in c.paragraphs:
                    pp.paragraph_format.keep_with_next = True
    para(doc, "", size=4, space_after=0, line=0.6)
    return t


def add_page_number_footer(section):
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    fld_begin = OxmlElement("w:fldChar"); fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText"); instr.set(qn("xml:space"), "preserve"); instr.text = " PAGE "
    fld_end = OxmlElement("w:fldChar"); fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin); run._r.append(instr); run._r.append(fld_end)
    set_font(run, SERIF, SONG, 9, False, (0x66, 0x6E, 0x78))


# --------------------------------------------------------------------------
# document
# --------------------------------------------------------------------------
def build():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.top_margin, sec.bottom_margin = Cm(1.9), Cm(1.8)
    sec.left_margin, sec.right_margin = Cm(2.15), Cm(2.15)
    add_page_number_footer(sec)

    style = doc.styles["Normal"]
    style.font.name = SERIF
    style.font.size = Pt(10.5)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), SONG)

    # ================= page 1: cover =================
    para(doc, "《MATLAB 基础与智能算法》期末大作业", size=10, align=WD_ALIGN_PARAGRAPH.CENTER,
         space_after=2, color=(0x66, 0x6E, 0x78))
    para(doc, "零工具箱 MATLAB 中手写实现并优化一个 8M 参数 Transformer 的 CPU 训练",
         size=17, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, ascii_font=HEI, ea_font=HEI,
         space_before=2, space_after=3, line=1.15)
    para(doc, "Hand-written forward/backward without any toolbox, and where the time actually goes",
         size=9.5, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=6, color=(0x55, 0x5F, 0x6B))

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(8)
    set_font(p.add_run(NUM["author"]), SERIF, SONG, 10, False, (0x22, 0x2A, 0x33))

    h2(doc, "摘要要点", space_before=2)
    hl = NUM["highlights"]
    for i, item in enumerate(hl, 1):
        bullet(doc, [(f"{i}  ", True), (item, False)], size=9.5, space_after=1.5)

    figure(doc, "fig1_cover.png", 16.5,
           "图 1  左：同一个训练步在六种实现下的耗时，括号内为相对基准写法的加速比。"
           "右：第 10 000 步模型对未见过提示的实际回复（温度 0.7，top-k 50）。")

    h2(doc, "核心结论", space_before=2)
    para(doc, NUM["oneline"], size=10, indent=0.0, space_after=5, line=1.3)

    table(doc,
          ["关键数字", "值", "关键数字", "值"],
          NUM["kpi_rows"],
          [3.4, 4.6, 3.4, 4.6], size=8.6)

    # ================= body =================
    doc.add_page_break()

    h1(doc, "1  问题与动机", space_before=0)
    h2(doc, "1.1  环境约束与实现前提")
    rich(doc, [("本机安装的 MATLAB R2026b 在 ", False),
               ("toolbox/", False, (0x1F, 0x4E, 0x79)),
               (" 下仅有 26 个目录，其中不含任何与深度学习或数值优化相关的工具箱：", False)],
         size=10.2, indent=0.74)
    para(doc, "缺少 Deep Learning Toolbox，因此没有 dlarray、dlgradient 与 trainnet，也就没有自动微分；"
              "缺少 Parallel Computing Toolbox，因此没有 gpuArray，也没有 parfor 的实际并行；"
              "缺少 Statistics、Optimization、Signal、Image 与 Symbolic，连 softmax（它位于 nnet 中）"
              "都无法直接调用，梯度检验也没有现成函数。", size=10.2, indent=0.74, space_after=4)
    para(doc, "因此手写前向与反向传播并非技术路线上的取舍，而是该环境下唯一可行的实现方式。"
              "这也使题目转化为一个更有意义的问题：一门以矩阵运算见长的语言，在去除所有深度学习封装之后，"
              "能够把一个 transformer 训练到什么程度？", size=10.2, indent=0.74, space_after=4)

    h2(doc, "1.2  研究问题与参照实现")
    para(doc, "参照实现选用 GitHub 上的 arman-bd/guppylm（MIT 协议）：一个刻意为小的 vanilla GPT，"
              "6 层、d=384、6 头、FFN 768、权重捆绑、学习式位置编码，原实现在 T4 上约 5 分钟完成训练。"
              "选择它有两个原因：其规模恰好落在可在 CPU 上完成训练的量级；且存在明确的 PyTorch 参考实现，"
              "使“实现是否正确”可以被客观检验，而不必依赖损失是否下降来判断。",
         size=10.2, indent=0.74, space_after=4)
    para(doc, "本文回答三个问题：① 仅具备基础 MATLAB 时，该模型能否被完整地训练出来；"
              "② 一个训练步的时间究竟消耗在哪里；③ 在 8 核 CPU 上，这条路径的性能上限由什么决定。",
         size=10.2, indent=0.74, space_after=4)

    h1(doc, "2  开发环境与工具")
    table(doc,
          ["项目", "配置"],
          NUM["env_rows"],
          [3.6, 12.6], caption="表 1  实验环境", size=8.8)
    para(doc, "辅助工具仅有 Python 3.13（numpy / pyarrow / tokenizers），用途严格限定于数据准备："
              "读取 parquet、训练 byte-level BPE、将 57 000 条对话一次性编码为两个 uint16 矩阵。"
              "此后训练与推理全程只处理整数 id，MATLAB 侧不参与分词——"
              "这也是训练循环中不含任何字符串操作的原因。",
         size=10.2, indent=0.74, space_after=4)
    para(doc, NUM["ai_note"], size=10.2, indent=0.74, space_after=4)

    h1(doc, "3  实现")
    h2(doc, "3.1  内存布局")
    para(doc, "PyTorch 的 Linear 计算 xWᵀ，权重按 [out, in] 存储。照搬到 MATLAB，每个通用矩阵乘（GEMM）都需要额外一次转置。"
              "本文改为按 [in, out] 存储，前向即 A*W：在列主序下 A 的行连续、W 的列连续，"
              "两个操作数都处于 BLAS（MATLAB 底层的线性代数库）最有利的布局，整条链路无需任何转置。由于这一步的吞吐完全由 BLAS 决定，后文的性能分析也以它的实测上限作为基准。",
         size=10.2, indent=0.74, space_after=4)
    para(doc, "第二个决定是激活的排布：将 [T, B, d] 压成 [T·B, d]，行号 n = (b−1)T + t。"
              "该顺序恰好等于 MATLAB 对 ids[T,B] 取 ids(:) 时的列主序顺序，因此 embedding 查表可直接写成 "
              "P.te(ids(:)+1, :)，既不需要 permute，也不需要将 gather 模拟为 one-hot 乘矩阵。",
         size=10.2, indent=0.74, space_after=4)

    h2(doc, "3.2  反向传播")
    para(doc, "八个算子全部手写。表 2 列出核心公式，其中 dX 的形状与激活一致，dW 与权重一致。",
         size=10.2, indent=0.74, space_after=4)
    table(doc,
          ["算子", "前向", "反向"],
          NUM["bwd_rows"],
          [3.1, 5.9, 7.2], caption="表 2  手写反向传播的核心公式", size=8.6)

    h2(doc, "3.3  三个关键算子")
    para(doc, "融合 softmax 与交叉熵。该实现不构造 one-hot，而是将 −1 直接写入概率矩阵的目标位置；"
              "损失只取目标类的 logit，省去整个 [N×V] 的对数概率矩阵。",
         size=10.2, indent=0.74, space_after=3)
    code_block(doc, NUM["code_ce"])

    para(doc, "LayerNorm 的反向。遗漏式中的最后一项，是典型的“仍能训练但梯度不正确”的错误，"
              "在有限差分下会放大至 1e−3 量级。", size=10.2, indent=0.74, space_before=2, space_after=3)
    code_block(doc, NUM["code_ln"])

    para(doc, "AdamW 的参数更新。将 v 的偏差校正折入两个标量，每个参数数组减少一次整块乘法与一次块内除法，"
              "在 8.06 M 参数上即为两趟 32 MB 读写。", size=10.2, indent=0.74, space_before=2, space_after=3)
    code_block(doc, NUM["code_adamw"])

    h2(doc, "3.4  源码结构与运行方式")
    code_block(doc, NUM["tree"])
    para(doc, "训练与采样入口为 run_train_resume.m 与 run_sample.m，超参数集中在 +guppy/defaultConfig.m "
              "一处定义，训练、基准与出图读取同一份 cfg。由于在线许可无法连接 MathWorks（错误 5201），"
              "matlab -batch 不可用，即不存在独立的后台执行能力；各项实验改由一个基于文件的命令队列驱动："
              "向 agent/inbox/ 写入 .m 文件，运行中的会话按文件名顺序 evalc 执行，结果写入 agent/outbox/ "
              "并更新心跳文件。全部三十余次实验与一次无人值守的长时间训练均通过该队列完成，"
              "无需中断会话即可排队、查看进度与读取结果。",
         size=10.2, indent=0.74, space_after=4)

    h1(doc, "4  数值正确性")
    h2(doc, "4.1  三级验证协议")
    para(doc, "对 8 M 参数逐个扰动做梯度校验并不可行：三档步长会给出符号不同的结果。"
              "原因是 ReLU 在扰动下翻转符号，数值导数被拐点淹没——这不是精度问题，而是方法不适用。"
              "因此采用三级验证：算子级（固定随机输入、double 精度、h=1e−4，相对误差要求小于 1e−7）、"
              "整网方向导数（1408 维随机方向），以及端到端单批过拟合（损失必须降至 0.0000）。"
              "第三级能一次发现“方向正确但某个张量漏乘”的错误，因为存在此类遗漏时无法完成过拟合。",
         size=10.2, indent=0.74, space_after=3)
    table(doc,
          ["检查项", "方法", "相对误差 / 结果"],
          NUM["grad_rows"],
          [4.3, 6.6, 5.3], caption="表 3  正确性证据", size=8.6)
    para(doc, "算子级校验采用中心差分（double，h=1e−4），相对误差按 ‖g_num−g_ana‖/‖g_num‖ 定义；"
              "整网方向导数在 1408 维随机方向上给出 2.7e−7，说明整条链路的梯度方向一致，"
              "而不只是单个算子正确。",
         size=10.2, indent=0.74, space_after=4)

    h2(doc, "4.2  一个不产生报错的失效：交叉熵的线性索引")
    rich(doc, [("现象  ", True), ("损失由 7.8 平稳降至 5.8，曲线正常，但采样结果全部为高频虚词。", False)],
         size=10.2, indent=0.74)
    rich(doc, [("定位  ", True), ("整网有限差分无法对齐（三档步长给出 0.09 / 0.24 / −0.49，符号不一致），"
                                  "于是转至算子级单独检验 softmaxCE：解析梯度与数值梯度在非目标类上完全一致，"
                                  "仅在目标类上相差一个常数。", False)], size=10.2, indent=0.74)
    rich(doc, [("根因  ", True), ("lin = i·V + k 为行主序；在 MATLAB 列主序下，(i,k) 的线性下标应为 i + k·N。"
                                  "该写法不会产生报错，只是将 −1 加到了错误的 token 上，"
                                  "因此模型仍在收敛，但学习的是错误的目标。", False)], size=10.2, indent=0.74)
    rich(doc, [("修复  ", True), ("lin = (1:N)′ + tgt·N，算子级相对误差回到 1.4e−8。", False)],
         size=10.2, indent=0.74)
    para(doc, "这一问题的意义超出其本身：能够收敛并不等于正确，只有与独立参照（数值梯度）比较，"
              "才能确认实现是否正确。", size=10.2, indent=0.74, space_after=4)

    h1(doc, "5  结果与性能分析")
    h2(doc, "5.1  训练结果")
    figure(doc, "fig2_training.png", 14.4,
           "图 2  上：完整训练轨迹（对数纵轴）。下：同一份数据在第 2 000 步之后的放大。"
           "训练损失每 20 步采样一次，留出集 3 000 条样本每 250 步评估一次。")
    para(doc, NUM["train_text"], size=10.2, indent=0.74, space_after=4)
    table(doc,
          ["用户提示", "模型回复（温度 0.7，top-k 50，遇到 <|im_end|> 停止）"],
          NUM["sample_rows"],
          [4.7, 11.5], caption="表 4  第 10 000 步模型对固定提示集的回复", size=8.6,
          align_center=False)

    h2(doc, "5.2  训练效率")
    para(doc, "性能数据采用交错测量：所有配置在同一轮内各测一次，重复 3–6 轮取最小值，"
              "并将最小值的稳定性本身作为指标报告。原因是单次测量不可信——同一配置连测 4 轮，"
              "块耗时在 0.687–0.720 s 之间波动，足以淹没 10% 量级的改动。表 5 与图 3 均来自该协议。",
         size=10.2, indent=0.74, space_after=3)
    table(doc,
          ["配置", "做了什么", "s/step", "加速", "累计"],
          NUM["ladder_rows"],
          [3.4, 7.5, 1.9, 1.5, 1.5], caption="表 5  优化阶梯（交错测量，三轮取最小值）",
          size=8.4, left_cols=[1])
    figure(doc, "fig3_waterfall.png", 16.4,
           "图 3  各项改动带来的边际收益。灰柱为起点，绿柱为该配置相对上一配置节省的时间，红柱为最终配置；"
           "末两根绿柱仅剩 0.03–0.04 s。")

    h2(doc, "5.3  单步耗时的构成")
    para(doc, NUM["ceiling_text"], size=10.2, indent=0.74, space_after=4)
    figure(doc, "fig4_roofline.png", 13.4,
           "图 4  本机单精度 GEMM 的实测吞吐。虚线为 1024³ 方阵的上限 509 GFLOPS，"
           "其余七项为模型实际使用的矩阵形状。")

    h2(doc, "5.4  优化空间的边界")
    for item in NUM["negatives"]:
        bullet(doc, [(item[0] + "。 ", True), (item[1], False)], size=10.0, space_after=2.5)

    h2(doc, "5.5  推理的成本结构")
    para(doc, NUM["infer_text"], size=10.2, indent=0.74, space_after=3)
    figure(doc, "fig5_inference.png", 16.4,
           "图 5  左：单 token 前向 3.56 ms 的逐行归因，红色为每步重建却从不被读取的激活缓存。"
           "右：同一组权重下生成 110 个 token 的单步耗时。")
    para(doc, NUM["infer_text2"], size=10.2, indent=0.74, space_after=4)

    h2(doc, "5.6  批量解码")
    para(doc, NUM["batch_text"], size=10.2, indent=0.74, space_after=3)
    figure(doc, "fig6_batch.png", 16.4,
           "图 6  批量解码的实测曲线。左图虚线为“吞吐随 B 线性增长”的理想线，红虚线为 roofline 模型给出的"
           "预测折点 B=30；右图点线为 3.56 ms 的权重读取下限。")
    para(doc, NUM["batch_text2"], size=10.2, indent=0.74, space_after=4)

    h1(doc, "6  总结与展望")
    for item in NUM["conclusions"]:
        bullet(doc, [(item[0] + "： ", True), (item[1], False)], size=10.0, space_after=2.5)
    para(doc, NUM["outlook"], size=10.2, indent=0.74, space_before=3, space_after=4)

    doc.add_page_break()
    h1(doc, "附录 A  复现清单", space_before=0)
    para(doc, "超参数只在 +guppy/defaultConfig.m 一处定义，训练、基准与出图读取同一份 cfg，"
              "因此文中所有数字对应同一套配置。",
         size=10.2, indent=0.74, space_after=3)
    table(doc,
          ["步骤", "入口", "产物 / 实测耗时"],
          NUM["appendix_rows"],
          [2.7, 8.4, 5.1], caption="表 6  复现清单", size=8.2, left_cols=[1, 2],
          keep_together=True)
    h1(doc, "附录 B  逐元素算子的实测成本")
    para(doc, NUM["prim_text"], size=10.2, indent=0.74, space_after=3)
    table(doc,
          ["表达式", "耗时 (ms)", "每元素 (ns)", "有效带宽 (GB/s)"],
          NUM["prim_rows"],
          [5.0, 3.6, 3.6, 4.0], caption="表 7  一次算子调用的实测成本", size=8.4, left_cols=[0],
          keep_together=True)
    para(doc, NUM["stream_text"], size=10.2, indent=0.74, space_before=4, space_after=4)

    para(doc, NUM["closing"], size=10.2, indent=0.74, space_before=4, space_after=4)

    out = OUT / "MATLAB大作业_零工具箱Transformer.docx"
    doc.save(str(out))
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    build()
