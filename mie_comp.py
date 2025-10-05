#MIT License

#Copyright (c) 2025 saysaysx

import numpy as np
import matplotlib.pyplot as plt
from math import pi
import PyMieScatt as ps
import pandas as pd


from math import *
import numpy

class optparsct:
    """
    Структура данных для хранения оптических параметров рассеяния:
    - sct: полное сечение рассеяния (см²)
    - ext: полное сечение поглощения + рассеяния (см²)
    - spi: интенсивность обратного рассеяния (в 180°), вычисленная через нормированное угловое распределение
    - spi1: коэффициент обратного рассеяния, вычисленный напрямую через Q_back = |Σ(a_n - b_n)(-1)^n(2n+1)|² / x²
    - ind: нормированное угловое распределение интенсивности I(θ)
    - sctang: дифференциальное сечение рассеяния dσ/dΩ = σ_sca * I(θ)
    """

    def __init__(self,n):
        self.sct = 0.0
        self.ext = 0.0
        self.spi = 0.0
        self.ind = numpy.zeros(n)    # I(θ) — нормированная интенсивность
        self.sctang = numpy.zeros(n) # dσ/dΩ
        self.spi1 = 0.0              # Q_back, вычисленный аналитически

def miesctopt(refrp=1+0j,refrm=1+0j,nang = 90,ang= [],r = 1.0,lam = 1.0):
    """
    Реализация теории Ми на чистом NumPy.

    Параметры:
    - refrp: показатель преломления частицы (вещественный или комплексный)
    - refrm: показатель преломления среды (обычно 1.0 для воздуха)
    - nang: количество углов для расчёта I(θ)
    - ang: массив углов рассеяния в радианах [0, π]
    - r: радиус частицы (в микрометрах)
    - lam: длина волны (в микрометрах)

    Возвращает: объект optparsct с рассчитанными параметрами.
    """

    opt = optparsct(nang)
    # Размерный параметр x = 2πr/λ
    x = 2.0*pi*r/lam

    # Оценка максимального номера парциальной волны (Bohren & Huffman)
    nmax = int(x+4.0*pow(x,1.0/3.0)+2.0)
    # Комплексный относительный показатель преломления
    m = complex(refrp/refrm)
    mx = complex(x*m)
    # Расчёт цепной дроби для коэффициентов D_n (алгоритм Lentz)
    nd = int(max(nmax,numpy.abs(mx))+3)+20 # запас для сходимости
    D = numpy.array([complex((i+1)/mx) for i in range(nd+1)])
    D[nd] = 0.0+0.0j
    for i in range(nd,0,-1):
        D[i - 1] = D[i - 1] - 1.0 / (D[i] + D[i - 1])

    # Рекуррентный расчёт сферических функций Бесселя (psi_n = x j_n(x))
    psi = numpy.zeros(nd+1)
    ksi = numpy.zeros(nd+1) # ksi_n = x h_n^(1)(x) — сферические функции Ханкеля
    psi[0]=sin(x)
    psi[1]=sin(x)/x-cos(x)
    ksi[0]=cos(x)
    ksi[1]=cos(x)/x+sin(x)
    for i in range(2,nd+1):
        psi[i]=((2.0*(i-1)+1.0)*psi[i-1]/x)-psi[i-2]
        ksi[i]=((2.0*(i-1)+1.0)*ksi[i-1]/x)-ksi[i-2]

    # eps_n = psi_n - i * ksi_n = x * h_n^(1)(x)
    eps = numpy.array([complex(psi[i],-ksi[i]) for i in range(nd+1)])
    # Коэффициенты Ми a_n и b_n
    ari = numpy.arange(nd+1)
    arix = ari/x
    tmp1 = D/m+arix
    tmp2 = D*m+arix
    psis = numpy.roll(psi,1)   # psi_{n-1}
    epss =  numpy.roll(eps,1)  # eps_{n-1}
    a = (tmp1*psi - psis)/(tmp1*eps-epss)  # электрические мультиполи
    b = (tmp2*psi - psis)/(tmp2*eps-epss)  # магнитные мультиполи

    # Подготовка полиномов Лежандра для углового распределения
    ar2i1 = numpy.arange(nd+1)*2+1 # (2n+1)  #numpy.array([2*i+1 for i in range(nd+1)])
    ar2i1i =ari*(ari+1) # n(n+1) #numpy.array([(i+1)*i for i in range(nd+1)])
    ar2i1i[0] = 1  # избегаем деления на 0
    pqq = numpy.zeros(shape = (nang,nd+1))  # P_n(cosθ)
    pqs = numpy.zeros(shape = (nang,nd+1))  # P_{n-1}(cosθ)
    tqq = numpy.zeros(shape = (nang,nd+1))  # τ_n = dP_n/dθ
    pqq[:,1] = 1.0
    cosq = numpy.array([[cos(ang[i]) for i in range(nang)]])
    for i in range(2,nd+1):
        pqq[:,i] = ((2.0*i-1.0)/(i-1.0))*cosq[0,:]*pqq[:,i-1]-i*pqq[:,i-2]/(i-1.0)
    pqs[:,1:-1] = pqq[:,0:-2]#numpy.roll(pq,1)
    tqq = cosq.T*pqq*ari- pqs*(ari+1)

    # Нормировочные коэффициенты для S1 и S2
    ardiv = ar2i1/ar2i1i

    # Амплитуды рассеяния S1(θ) и S2(θ)
    ss1 = (a*pqq+b*tqq)*ardiv
    ss2 = (a*tqq+b*pqq)*ardiv
    s1 = ss1.sum(axis=1)
    s2 = ss2.sum(axis=1)

    # Расчёт эффективных сечений
    ar1 = (-1)**ari # (-1)^n
    qext = ((a+b).real*ar2i1).sum() # Q_ext * x²/2
    qsct = ((abs(a)**2+abs(b)**2)*ar2i1).sum()  # Q_sca * x²/2
    qsctpic = ((a-b)*ar1*ar2i1).sum() # Σ(a_n - b_n)(-1)^n(2n+1)

    # Перевод в стандартные эффективные коэффициенты
    qsctpi = abs(qsctpic)**2/(x**2)  # Q_back
    qext=qext*2.0/x**2.0
    qsct=qsct*2.0/x**2.0

    # Перевод в физические сечения (см²), с учётом масштаба 1e-8 (мкм² → см²)
    opt.sct=qsct*(r**2)*pi*1e-8
    opt.ext=qext*(r**2)*pi*1e-8
    opt.spi1 = qsctpi*(r**2)*pi*1e-8

    # Угловое распределение интенсивности
    opt.ind= abs(s1)**2+abs(s2)**2

    # Нормировка I(θ): ∫ I(θ) sinθ dθ = 1
    ind1 = numpy.roll(opt.ind,-1)
    sinang = numpy.sin(ang)
    sinang1 = numpy.roll(sinang,-1)
    ang1 = numpy.roll(ang,-1)
    dang = ang1-ang
    dang[-1] = 0
    inds = ((sinang*opt.ind+sinang1*ind1)*dang).sum()*0.5
    opt.ind = opt.ind/inds

    # Интенсивность в 180° (обратное рассеяние)
    opt.spi=opt.sct*opt.ind[nang-1]*2 # dσ/dΩ в 180°  #qsctpi*(r**2)*pi*1e-8#/(4*pi)
    opt.sctang = opt.sct * opt.ind*2 # полное дифференциальное сечение

    return opt


def compare_with_pymiescatt1():
    """
    Сравнение собственной реализации с библиотекой PyMieScatt.
    Проверяются:
    - Q_sca, Q_ext — эффективные коэффициенты рассеяния и экстинкции
    - Q_back — коэффициент обратного рассеяния
    - I(θ) — нормированное угловое распределение

    Визуализация:
    - Линейный и логарифмический графики I(θ)
    - Столбчатая диаграмма коэффициентов
    """
    # Параметры
    wavelength_nm = 550.0     # нм
    radius_um = 0.3           # мкм
    radius_nm = radius_um * 1000
    diameter_nm = 2 * radius_nm
    diameter_um = 2 * radius_um
    n_particle = 1.59 + 0.0j  # комплексный показатель преломления
    n_medium = 1.0            # среда — воздух

    nang = 180
    angles_deg = np.linspace(0, 180, nang)
    angles_rad = np.deg2rad(angles_deg)

    print("=== Сравнение с PyMieScatt ===")

    # === ВАША РЕАЛИЗАЦИЯ ===
    result_yours = miesctopt(
        refrp=n_particle.real,
        refrm=n_medium,
        nang=nang,
        ang=angles_rad,
        r=radius_um,
        lam=wavelength_nm / 1000.0
    )

    factor = pi * radius_um**2 #* 1e-8
    Q_sca_yours = result_yours.sct / factor * 1e8
    Q_ext_yours = result_yours.ext / factor * 1e8
    Q_back_yours_spi = result_yours.spi / factor * 1e8      # через opt.spi
    Q_back_yours_spi1 = result_yours.spi1 / factor * 1e8    # через opt.spi1
    I_yours = result_yours.ind

    print(f"Ваша реализация:")
    print(f"  Q_sca       = {Q_sca_yours:.6f}")
    print(f"  Q_ext       = {Q_ext_yours:.6f}")
    print(f"  Q_back (spi) = {Q_back_yours_spi:.6e}")
    print(f"  Q_back (spi1)= {Q_back_yours_spi1:.6e}")

    # === PYMIESCATT ===
    qext_pymie, qsca_pymie, qabs_pymie, *_ = ps.MieQ(
        m=n_particle,
        wavelength=wavelength_nm,
        diameter=diameter_nm,
        nMedium=n_medium
    )

    # Получаем qback — коэффициент обратного рассеяния
    # (в PyMieScatt он возвращается как 6-й элемент)
    result_pymie = ps.MieQ(m=n_particle, wavelength=wavelength_nm, diameter=diameter_nm, nMedium=n_medium)
    if len(result_pymie) >= 6:
        qext_pymie, qsca_pymie, qabs_pymie, g, qpr, qback_pymie = result_pymie[:6]
    else:
        qback_pymie = 0.0  # fallback

    # Угловое распределение — для графика
    wavelength_um = wavelength_nm / 1000.0
    x = np.pi * diameter_um / wavelength_um
    mu = np.cos(angles_rad)

    S1_list = []
    S2_list = []
    for mu_val in mu:
        s1_val, s2_val = ps.MieS1S2(m=n_particle, x=x, mu=mu_val)
        S1_list.append(s1_val)
        S2_list.append(s2_val)

    S1 = np.array(S1_list)
    S2 = np.array(S2_list)
    intensity_pymie = (np.abs(S1)**2 + np.abs(S2)**2)

    # Нормируем интенсивность (как у вас)
    sin_theta = np.sin(angles_rad)
    if hasattr(np, 'trapezoid'):
        trapz_func = np.trapezoid
    else:
        trapz_func = np.trapz
    integral = trapz_func(intensity_pymie * sin_theta, angles_rad)
    I_pymie = intensity_pymie / integral

    print(f"\nPyMieScatt:")
    print(f"  Q_sca   = {qsca_pymie:.6f}")
    print(f"  Q_ext   = {qext_pymie:.6f}")
    print(f"  Q_back  = {qback_pymie:.6e}")

    # === Сравнение коэффициентов ===
    print(f"\n=== Расхождения ===")
    delta_Q_sca = abs(Q_sca_yours - qsca_pymie)
    delta_Q_ext = abs(Q_ext_yours - qext_pymie)
    delta_Q_back_spi = abs(Q_back_yours_spi - qback_pymie)
    delta_Q_back_spi1 = abs(Q_back_yours_spi1 - qback_pymie)
    max_delta_I = np.max(np.abs(I_yours - I_pymie))

    print(f"  ΔQ_sca          = {delta_Q_sca:.3e}")
    print(f"  ΔQ_ext          = {delta_Q_ext:.3e}")
    print(f"  ΔQ_back (spi)   = {delta_Q_back_spi:.3e}")
    print(f"  ΔQ_back (spi1)  = {delta_Q_back_spi1:.3e}")
    print(f"  Макс. ΔI(θ)     = {max_delta_I:.3e}")

    # === Графики ===
    fig = plt.figure(figsize=(18, 5))

    # График 1: Угловое распределение
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(angles_deg, I_yours, label='Ваша реализация', linewidth=3, alpha=0.7)
    ax1.plot(angles_deg, I_pymie, '--', label='PyMieScatt', linewidth=2, color='red')
    ax1.set_xlabel('Угол рассеяния (градусы)')
    ax1.set_ylabel('Нормированная интенсивность')
    ax1.set_title('Интенсивность рассеяния I(θ)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # График 2: Логарифмический масштаб
    ax2 = plt.subplot(1, 3, 2)
    ax2.semilogy(angles_deg, I_yours, label='Ваша реализация', linewidth=3, alpha=0.7)
    ax2.semilogy(angles_deg, I_pymie, '--', label='PyMieScatt', linewidth=2, color='red')
    ax2.set_xlabel('Угол рассеяния (градусы)')
    ax2.set_ylabel('Интенсивность (лог. масштаб)')
    ax2.set_title('I(θ) — логарифмический масштаб')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # График 3: Сравнение коэффициентов — столбчатая диаграмма
    ax3 = plt.subplot(1, 3, 3)
    labels = ['Q_sca', 'Q_ext', 'Q_back']
    your_values = [Q_sca_yours, Q_ext_yours, Q_back_yours_spi]
    pymie_values = [qsca_pymie, qext_pymie, qback_pymie]

    x = np.arange(len(labels))
    width = 0.35

    ax3.bar(x - width/2, your_values, width, label='Ваша реализация', alpha=0.7)
    ax3.bar(x + width/2, pymie_values, width, label='PyMieScatt', alpha=0.7)

    ax3.set_ylabel('Значение коэффициента')
    ax3.set_title('Сравнение коэффициентов')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        'Q_sca_yours': Q_sca_yours,
        'Q_ext_yours': Q_ext_yours,
        'Q_back_yours_spi': Q_back_yours_spi,
        'Q_back_yours_spi1': Q_back_yours_spi1,
        'Q_sca_pymie': qsca_pymie,
        'Q_ext_pymie': qext_pymie,
        'Q_back_pymie': qback_pymie,
        'delta_Q_sca': delta_Q_sca,
        'delta_Q_ext': delta_Q_ext,
        'delta_Q_back_spi': delta_Q_back_spi,
        'delta_Q_back_spi1': delta_Q_back_spi1,
        'max_delta_I': max_delta_I
    }


def compare_with_pymiescatt2():
    """
    Сравниваем вашу реализацию с PyMieScatt по Q_sca, Q_ext, Q_back.
    Визуализируем угловое распределение + точку обратного рассеяния + отдельные графики для spi/spi1/qback.
    """
    # Параметры
    wavelength_nm = 550.0     # нм
    radius_um = 0.3           # мкм
    radius_nm = radius_um * 1000.0
    diameter_nm = 2 * radius_nm  # ← PyMieScatt ожидает диаметр в НАНОМЕТРАХ
    n_particle = 1.59 + 0.0j  # комплексный показатель преломления
    n_medium = 1.0            # среда — воздух

    nang = 180
    angles_deg = np.linspace(0, 180, nang)
    angles_rad = np.deg2rad(angles_deg)

    print("=== Сравнение с PyMieScatt ===")

    # === ВАША РЕАЛИЗАЦИЯ ===
    result_yours = miesctopt(
        refrp=n_particle.real,
        refrm=n_medium,
        nang=nang,
        ang=angles_rad,
        r=radius_um,
        lam=wavelength_nm / 1000.0
    )

    factor = pi * radius_um**2  # площадь частицы (мкм²)
    Q_sca_yours = result_yours.sct / factor * 1e8
    Q_ext_yours = result_yours.ext / factor * 1e8
    Q_back_yours_spi = result_yours.spi / factor * 1e8      # через opt.spi
    Q_back_yours_spi1 = result_yours.spi1 / factor * 1e8    # через opt.spi1
    I_yours = result_yours.ind

    print(f"Ваша реализация:")
    print(f"  Q_sca       = {Q_sca_yours:.6f}")
    print(f"  Q_ext       = {Q_ext_yours:.6f}")
    print(f"  Q_back (spi) = {Q_back_yours_spi:.6e}")
    print(f"  Q_back (spi1)= {Q_back_yours_spi1:.6e}")

    # === PYMIESCATT ===
    # ✅ diameter в НАНОМЕТРАХ — как требует документация
    result_pymie = ps.MieQ(
        m=n_particle,
        wavelength=wavelength_nm,
        diameter=diameter_nm,  # ← ИСПРАВЛЕНО: в нанометрах
        nMedium=n_medium
    )

    if len(result_pymie) >= 6:
        qext_pymie, qsca_pymie, qabs_pymie, g, qpr, qback_pymie = result_pymie[:6]
    else:
        qback_pymie = 0.0

    # Угловое распределение
    # Для MieS1S2 — тоже diameter в нанометрах
    x = np.pi * diameter_nm / wavelength_nm  # x = π * d / λ (d и λ в одинаковых единицах — нм)
    mu = np.cos(angles_rad)

    S1_list = []
    S2_list = []
    for mu_val in mu:
        s1_val, s2_val = ps.MieS1S2(m=n_particle, x=x, mu=mu_val)
        S1_list.append(s1_val)
        S2_list.append(s2_val)

    S1 = np.array(S1_list)
    S2 = np.array(S2_list)
    intensity_pymie_unscaled = np.abs(S1)**2 + np.abs(S2)**2

    # Нормируем (как у вас)
    sin_theta = np.sin(angles_rad)
    trapz_func = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    integral = trapz_func(intensity_pymie_unscaled * sin_theta, angles_rad)
    I_pymie = intensity_pymie_unscaled / integral

    # Значение в 180°
    I_yours_180 = I_yours[-1]
    I_pymie_180 = I_pymie[-1]

    # Пересчитываем spi1 в интенсивность в 180° для графика
    I_spi1_180 = Q_back_yours_spi1 / Q_sca_yours if Q_sca_yours != 0 else 0.0

    print(f"\nPyMieScatt:")
    print(f"  Q_sca   = {qsca_pymie:.6f}")
    print(f"  Q_ext   = {qext_pymie:.6f}")
    print(f"  Q_back  = {qback_pymie:.6e}")
    print(f"  I(180°) = {I_pymie_180:.6e}")

    # === Сравнение ===
    delta_Q_sca = abs(Q_sca_yours - qsca_pymie)
    delta_Q_ext = abs(Q_ext_yours - qext_pymie)
    delta_Q_back_spi = abs(Q_back_yours_spi - qback_pymie)
    delta_Q_back_spi1 = abs(Q_back_yours_spi1 - qback_pymie)
    max_delta_I = np.max(np.abs(I_yours - I_pymie))

    print(f"\n=== Расхождения ===")
    print(f"  ΔQ_sca          = {delta_Q_sca:.3e}")
    print(f"  ΔQ_ext          = {delta_Q_ext:.3e}")
    print(f"  ΔQ_back (spi)   = {delta_Q_back_spi:.3e}")
    print(f"  ΔQ_back (spi1)  = {delta_Q_back_spi1:.3e}")
    print(f"  Макс. ΔI(θ)     = {max_delta_I:.3e}")

    # === Графики ===
    fig = plt.figure(figsize=(18, 12))

    # График 1: Угловое распределение
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(angles_deg, I_yours, label='Ваша реализация', linewidth=3, alpha=0.7)
    ax1.plot(angles_deg, I_pymie, '--', label='PyMieScatt', linewidth=2, color='red')
    ax1.plot(180, I_yours_180, 'bo', markersize=8, label='Ваше I(180°) из spi')
    ax1.plot(180, I_pymie_180, 'ro', markersize=8, label='PyMieScatt I(180°)')
    ax1.plot(180, I_spi1_180, 'go', markersize=8, label='Ваше I(180°) из spi1')
    ax1.set_xlabel('Угол рассеяния (градусы)')
    ax1.set_ylabel('Нормированная интенсивность')
    ax1.set_title('Интенсивность рассеяния I(θ)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # График 2: Логарифмический масштаб
    ax2 = plt.subplot(2, 3, 2)
    ax2.semilogy(angles_deg, I_yours, label='Ваша реализация', linewidth=3, alpha=0.7)
    ax2.semilogy(angles_deg, I_pymie, '--', label='PyMieScatt', linewidth=2, color='red')
    ax2.plot(180, I_yours_180, 'bo', markersize=8)
    ax2.plot(180, I_pymie_180, 'ro', markersize=8)
    ax2.plot(180, I_spi1_180, 'go', markersize=8)
    ax2.set_xlabel('Угол рассеяния (градусы)')
    ax2.set_ylabel('Интенсивность (лог. масштаб)')
    ax2.set_title('I(θ) — логарифмический масштаб')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # График 3: Сравнение коэффициентов (Q_sca, Q_ext, Q_back)
    ax3 = plt.subplot(2, 3, 3)
    labels = ['Q_sca', 'Q_ext', 'Q_back']
    your_values = [Q_sca_yours, Q_ext_yours, Q_back_yours_spi]
    pymie_values = [qsca_pymie, qext_pymie, qback_pymie]
    x = np.arange(len(labels))
    width = 0.35
    ax3.bar(x - width/2, your_values, width, label='Ваша реализация', alpha=0.7)
    ax3.bar(x + width/2, pymie_values, width, label='PyMieScatt', alpha=0.7)
    ax3.set_ylabel('Значение коэффициента')
    ax3.set_title('Сравнение Q_sca, Q_ext, Q_back')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # График 4: Сравнение интенсивности в 180°
    ax4 = plt.subplot(2, 3, 4)
    back_labels = ['Ваше (spi)', 'Ваше (spi1)', 'PyMieScatt']
    back_values = [I_yours_180, I_spi1_180, I_pymie_180]
    bars4 = ax4.bar(back_labels, back_values, color=['blue', 'green', 'red'], alpha=0.7)
    ax4.set_ylabel('Интенсивность I(180°)')
    ax4.set_title('Сравнение интенсивности в 180°')
    ax4.grid(axis='y', alpha=0.3)
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + max(back_values)*0.01, f'{height:.2e}', ha='center', va='bottom')

    # График 5: Сравнение коэффициентов обратного рассеяния Q_back
    ax5 = plt.subplot(2, 3, 5)
    qback_labels = ['Ваше (spi)', 'Ваше (spi1)', 'PyMieScatt']
    qback_values = [Q_back_yours_spi, Q_back_yours_spi1, qback_pymie]
    bars5 = ax5.bar(qback_labels, qback_values, color=['blue', 'green', 'red'], alpha=0.7)
    ax5.set_ylabel('Q_back')
    ax5.set_title('Сравнение коэффициентов обратного рассеяния')
    ax5.grid(axis='y', alpha=0.3)
    for bar in bars5:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + max(qback_values)*0.01, f'{height:.2e}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    return {
        'Q_sca_yours': Q_sca_yours,
        'Q_ext_yours': Q_ext_yours,
        'Q_back_yours_spi': Q_back_yours_spi,
        'Q_back_yours_spi1': Q_back_yours_spi1,
        'Q_sca_pymie': qsca_pymie,
        'Q_ext_pymie': qext_pymie,
        'Q_back_pymie': qback_pymie,
        'I_180_yours': I_yours_180,
        'I_180_spi1': I_spi1_180,
        'I_180_pymie': I_pymie_180,
        'delta_Q_sca': delta_Q_sca,
        'delta_Q_ext': delta_Q_ext,
        'delta_Q_back_spi': delta_Q_back_spi,
        'delta_Q_back_spi1': delta_Q_back_spi1,
        'max_delta_I': max_delta_I
    }


def compare_multiple_parameters(wavelengths_nm, radii_um, nang=180):
    """
    Сравнивает вашу реализацию с PyMieScatt для нескольких длин волн и радиусов.
    Строит графики зависимостей Q_sca, Q_ext, Q_back от параметров.
    """
    results = []

    # Подготовка сетки углов (одинаковая для всех)
    angles_deg = np.linspace(0, 180, nang)
    angles_rad = np.deg2rad(angles_deg)

    print(f"=== Сравнение для {len(wavelengths_nm)} длин волн и {len(radii_um)} радиусов ===")

    for wl in wavelengths_nm:
        for r_um in radii_um:
            print(f"\n--- Длина волны: {wl} нм, Радиус: {r_um} мкм ---")

            # === ВАША РЕАЛИЗАЦИЯ ===
            result_yours = miesctopt(
                refrp=1.59,  # можно сделать параметром
                refrm=1.0,
                nang=nang,
                ang=angles_rad,
                r=r_um,
                lam=wl / 1000.0  # мкм
            )

            factor = pi * r_um**2
            Q_sca_yours = result_yours.sct / factor * 1e8
            Q_ext_yours = result_yours.ext / factor * 1e8
            Q_back_yours_spi = result_yours.spi / factor * 1e8
            Q_back_yours_spi1 = result_yours.spi1 / factor * 1e8

            # === PYMIESCATT ===
            diameter_nm = 2 * r_um * 1000  # в нанометрах
            result_pymie = ps.MieQ(
                m=1.59 + 0.0j,
                wavelength=wl,
                diameter=diameter_nm,
                nMedium=1.0
            )

            if len(result_pymie) >= 6:
                qext_pymie, qsca_pymie, qabs_pymie, g, qpr, qback_pymie = result_pymie[:6]
            else:
                qext_pymie = qsca_pymie = qback_pymie = 0.0

            # Сохраняем результаты
            results.append({
                'wavelength_nm': wl,
                'radius_um': r_um,
                'Q_sca_yours': Q_sca_yours,
                'Q_ext_yours': Q_ext_yours,
                'Q_back_yours_spi': Q_back_yours_spi,
                'Q_back_yours_spi1': Q_back_yours_spi1,
                'Q_sca_pymie': qsca_pymie,
                'Q_ext_pymie': qext_pymie,
                'Q_back_pymie': qback_pymie,
                'delta_Q_sca': abs(Q_sca_yours - qsca_pymie),
                'delta_Q_ext': abs(Q_ext_yours - qext_pymie),
                'delta_Q_back_spi': abs(Q_back_yours_spi - qback_pymie),
                'delta_Q_back_spi1': abs(Q_back_yours_spi1 - qback_pymie)
            })

    # Создаем DataFrame для удобства
    df = pd.DataFrame(results)
    print("\n=== Итоговая таблица расхождений ===")
    print(df[['wavelength_nm', 'radius_um', 'delta_Q_sca', 'delta_Q_ext', 'delta_Q_back_spi', 'delta_Q_back_spi1']])

    # === Графики ===
    fig = plt.figure(figsize=(18, 12))

    # График 1: Q_sca от радиуса (для каждой длины волны)
    ax1 = plt.subplot(2, 3, 1)
    for wl in wavelengths_nm:
        df_wl = df[df['wavelength_nm'] == wl]
        ax1.plot(df_wl['radius_um'], df_wl['Q_sca_yours'], 'o-', label=f'Ваше {wl}нм', alpha=0.7)
        ax1.plot(df_wl['radius_um'], df_wl['Q_sca_pymie'], 'x--', label=f'PyMie {wl}нм', alpha=0.7)
    ax1.set_xlabel('Радиус (мкм)')
    ax1.set_ylabel('Q_sca')
    ax1.set_title('Зависимость Q_sca от радиуса')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # График 2: Q_ext от радиуса
    ax2 = plt.subplot(2, 3, 2)
    for wl in wavelengths_nm:
        df_wl = df[df['wavelength_nm'] == wl]
        ax2.plot(df_wl['radius_um'], df_wl['Q_ext_yours'], 'o-', label=f'Ваше {wl}нм', alpha=0.7)
        ax2.plot(df_wl['radius_um'], df_wl['Q_ext_pymie'], 'x--', label=f'PyMie {wl}нм', alpha=0.7)
    ax2.set_xlabel('Радиус (мкм)')
    ax2.set_ylabel('Q_ext')
    ax2.set_title('Зависимость Q_ext от радиуса')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # График 3: Q_back от радиуса (через spi)
    ax3 = plt.subplot(2, 3, 3)
    for wl in wavelengths_nm:
        df_wl = df[df['wavelength_nm'] == wl]
        ax3.plot(df_wl['radius_um'], df_wl['Q_back_yours_spi'], 'o-', label=f'Ваше (spi) {wl}нм', alpha=0.7)
        ax3.plot(df_wl['radius_um'], df_wl['Q_back_yours_spi1'], 's-', label=f'Ваше (spi1) {wl}нм', alpha=0.7)
        ax3.plot(df_wl['radius_um'], df_wl['Q_back_pymie'], 'x--', label=f'PyMie {wl}нм', alpha=0.7)
    ax3.set_xlabel('Радиус (мкм)')
    ax3.set_ylabel('Q_back')
    ax3.set_title('Зависимость Q_back (spi) от радиуса')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # График 4: Q_sca от длины волны (для каждого радиуса)
    ax4 = plt.subplot(2, 3, 4)
    for r in radii_um:
        df_r = df[df['radius_um'] == r]
        ax4.plot(df_r['wavelength_nm'], df_r['Q_sca_yours'], 'o-', label=f'Ваше {r}мкм', alpha=0.7)
        ax4.plot(df_r['wavelength_nm'], df_r['Q_sca_pymie'], 'x--', label=f'PyMie {r}мкм', alpha=0.7)
    ax4.set_xlabel('Длина волны (нм)')
    ax4.set_ylabel('Q_sca')
    ax4.set_title('Зависимость Q_sca от длины волны')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # График 5: Q_back от длины волны
    ax5 = plt.subplot(2, 3, 5)
    for r in radii_um:
        df_r = df[df['radius_um'] == r]
        ax5.plot(df_r['wavelength_nm'], df_r['Q_back_yours_spi'], 'o-', label=f'Ваше (spi) {r}мкм', alpha=0.7)
        ax5.plot(df_r['wavelength_nm'], df_r['Q_back_pymie'], 'x--', label=f'PyMie {r}мкм', alpha=0.7)
    ax5.set_xlabel('Длина волны (нм)')
    ax5.set_ylabel('Q_back')
    ax5.set_title('Зависимость Q_back от длины волны')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # График 6: Максимальные расхождения
    ax6 = plt.subplot(2, 3, 6)
    metrics = ['delta_Q_sca', 'delta_Q_ext', 'delta_Q_back_spi']
    max_errors = [df[metric].max() for metric in metrics]
    ax6.bar(metrics, max_errors, color=['blue', 'orange', 'green'], alpha=0.7)
    ax6.set_ylabel('Макс. расхождение')
    ax6.set_title('Максимальные расхождения по метрикам')
    ax6.grid(axis='y', alpha=0.3)
    for i, v in enumerate(max_errors):
        ax6.text(i, v + max(max_errors)*0.01, f"{v:.1e}", ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    # Опционально: сохранить в Excel
    # df.to_excel('mie_comparison_results.xlsx', index=False)
    # print("\nРезультаты сохранены в 'mie_comparison_results.xlsx'")

    return df



class lam_refr:
    """
    Контейнер для параметров: длина волны и комплексные показатели преломления.
    Используется для векторизованных расчётов в TensorFlow.
    """
    def __init__(self, nlam):
        self.nlam = nlam
        self.lam = numpy.ones(nlam)
        self.refrp = numpy.ones(nlam, dtype=complex)
        self.refrm = numpy.ones(nlam, dtype=complex)
        self.refrm[:] = 1.0 + 0.0j

import tensorflow as tf

def miesct_tensor(rl=lam_refr(1), ang=numpy.array([0, pi/2]), r=numpy.array([1])):
    """
    Векторизованная реализация теории Ми на TensorFlow.
    Поддерживает:
    - несколько длин волн (nlam)
    - несколько радиусов (nr)
    - несколько углов рассеяния (nang)

    Особенности:
    - Все операции тензорные → позволяет использовать GPU
    - Алгоритм повторяет логику NumPy-версии, но с tf.Variable и tf.assign
    - Используется та же рекуррентная схема для psi, ksi, D_n

    Возвращает:
    - ext, sct: сечения экстинкции и рассеяния
    - spi: интенсивность в 180° через нормированное I(θ)
    - sctang: дифференциальное сечение dσ/dΩ(θ)
    - spi1: Q_back, вычисленный напрямую
    """
    nlam = rl.nlam
    nr = r.shape[0]
    nang = ang.shape[0]

    # Приведение к тензорам TensorFlow
    lam = tf.cast(rl.lam, tf.float64)
    r = tf.cast(r, tf.float64)
    refrp = tf.cast(rl.refrp, tf.complex128)
    refrm = tf.cast(rl.refrm, tf.complex128)

    # Размерный параметр x = 2πr/λ для всех комбинаций (nlam × nr)
    x = 2.0 * pi * r[None, :] / lam[:, None]

    # Оценка максимального n (с запасом)
    nmax = tf.reduce_max(x + 4.0 * x**(1.0/3.0) + 2.0)
    nmax = tf.cast(nmax, tf.int32)

    m = refrp / refrm
    mx = tf.cast(x, tf.complex128) * m[:, None]

    # Определение глубины суммирования nd
    amx = tf.math.abs(mx)
    maxmx = tf.cast(tf.reduce_max(amx), tf.int32)
    nd = int(tf.maximum(nmax, maxmx) / 2) + 2  # эмпирический запас

    # Инициализация цепной дроби D_n
    index = tf.cast(tf.range(0, nd+1), tf.complex128)
    D = tf.Variable((index + 1) / mx[:, :, None])  # shape: (nlam, nr, nd+1)
    D[:, :, nd].assign(tf.zeros([nlam, nr], tf.complex128))

    # Обратный проход для вычисления D_n (алгоритм Lentz)
    for i in range(nd, 0, -1):
        D[:, :, i - 1].assign(D[:, :, i - 1] - 1.0 / (D[:, :, i] + D[:, :, i - 1]))

    # Рекуррентный расчёт psi_n и ksi_n
    psi = tf.Variable(tf.zeros([nlam, nr, nd+1], tf.float64))
    ksi = tf.Variable(tf.zeros([nlam, nr, nd+1], tf.float64))

    psi[:, :, 0].assign(tf.math.sin(x))
    psi[:, :, 1].assign(tf.math.sin(x) / x - tf.math.cos(x))
    ksi[:, :, 0].assign(tf.math.cos(x))
    ksi[:, :, 1].assign(tf.math.cos(x) / x + tf.math.sin(x))

    for i in range(2, nd+1):
        psi[:, :, i].assign(((2.0*(i-1)+1.0) * psi[:, :, i-1] / x) - psi[:, :, i-2])
        ksi[:, :, i].assign(((2.0*(i-1)+1.0) * ksi[:, :, i-1] / x) - ksi[:, :, i-2])

    eps = tf.complex(psi, -ksi)

    # Коэффициенты Ми a_n, b_n
    a = tf.Variable(tf.zeros([nlam, nr, nd+1], tf.complex128))
    b = tf.Variable(tf.zeros([nlam, nr, nd+1], tf.complex128))
    psic = tf.cast(psi, tf.complex128)

    ix = index[:, None, None] / tf.cast(x, tf.complex128)
    ix = tf.transpose(ix, perm=(1, 2, 0))  # (nlam, nr, nd+1)

    tmp1 = D / m[:, None, None] + ix
    tmp2 = D * m[:, None, None] + ix

    # Вычисление a_n, b_n для n ≥ 1
    a[:, :, 1:nd+1].assign(
        (tmp1[:, :, 1:] * psic[:, :, 1:] - psic[:, :, :-1]) /
        (tmp1[:, :, 1:] * eps[:, :, 1:] - eps[:, :, :-1])
    )
    b[:, :, 1:nd+1].assign(
        (tmp2[:, :, 1:] * psic[:, :, 1:] - psic[:, :, :-1]) /
        (tmp2[:, :, 1:] * eps[:, :, 1:] - eps[:, :, :-1])
    )

    # Полиномы Лежандра и τ_n
    pq = tf.Variable(tf.zeros([nlam, nr, nang, nd+1], tf.complex128))
    tq = tf.Variable(tf.zeros([nlam, nr, nang, nd+1], tf.complex128))

    pq[:, :, :, 1].assign(1.0)
    cos_ang = tf.cast(tf.math.cos(ang), tf.complex128)
    tq[:, :, :, 1].assign(cos_ang)

    for i in range(2, nd+1):
        pq[:, :, :, i].assign(
            ((2.0*i - 1.0) / (i - 1.0)) * tq[:, :, :, 1] * pq[:, :, :, i-1] - (i / (i - 1.0)) * pq[:, :, :, i-2]
        )
        tq[:, :, :, i].assign(
            i * tq[:, :, :, 1] * pq[:, :, :, i] - (i + 1.0) * pq[:, :, :, i-1]
        )

    # Амплитуды S1, S2
    s1a = tf.Variable(tf.zeros([nlam, nr, nang, nd+1], tf.complex128))
    s2a = tf.Variable(tf.zeros([nlam, nr, nang, nd+1], tf.complex128))

    for i in range(1, nd+1):
        iv = (2.0 * i + 1.0) / ((i + 1) * i)
        s1a[:, :, :, i].assign(iv * (a[:, :, i, None] * pq[:, :, :, i] + b[:, :, i, None] * tq[:, :, :, i]))
        s2a[:, :, :, i].assign(iv * (a[:, :, i, None] * tq[:, :, :, i] + b[:, :, i, None] * pq[:, :, :, i]))

    s1 = tf.reduce_sum(s1a, axis=3)
    s2 = tf.reduce_sum(s2a, axis=3)

    # Эффективные коэффициенты
    indexf = tf.cast(tf.range(0, nd+1), tf.float64)
    qexta = tf.math.real((a + b) * (2 * index + 1))
    qext = tf.reduce_sum(qexta, axis=2)
    qscta = (tf.math.abs(a)**2 + tf.math.abs(b)**2) * (2.0 * indexf + 1.0)
    qsct = tf.reduce_sum(qscta, axis=2)
    qsctpica = (a - b) * ((-1) ** index) * (2 * index + 1)
    qsctpi = tf.abs(tf.reduce_sum(qsctpica, axis=2))**2 / (x**2)

    qext = qext * 2.0 / x**2.0
    qsct = qsct * 2.0 / x**2.0

    # Физические сечения (см²)
    sct = qsct * (r**2) * pi * 1e-8
    ext = qext * (r**2) * pi * 1e-8

    # Угловое распределение
    ind = tf.math.abs(s1)**2 + tf.math.abs(s2)**2
    sinang = tf.math.sin(ang)

    # Численное интегрирование для нормировки
    inds = (ind[:, :, 1:] * sinang[1:] + ind[:, :, :-1] * sinang[:-1]) * 0.5 * (ang[1:] - ang[:-1])
    indsv = tf.reduce_sum(inds, axis=2)
    indnew = ind / indsv[:, :, None]

    spi = indnew[:, :, -1] * sct * 2  # в 180°
    sctang = indnew * sct[:, :, None] * 2
    spi1 = qsctpi * (r**2) * 1e-12  # альтернативный Q_back

    return ext, sct, spi, sctang, spi1


def mieosct_tensor(rl=lam_refr(1), r=numpy.array([1])):
    """
    Упрощённая версия miesct_tensor без углового распределения.
    Используется, когда нужны только интегральные параметры (Q_sca, Q_ext, Q_back).
    Экономит память и время при массовых расчётах.
    """
    # ... (аналогично, но без расчёта полиномов Лежандра и S1/S2)

    nlam = rl.nlam
    nr = r.shape[0]


    lam = tf.cast(rl.lam,tf.float64)
    r = tf.cast(r,tf.float64)
    refrp = tf.cast(rl.refrp,tf.complex128)
    refrm = tf.cast(rl.refrm,tf.complex128)
    # size nlam*nr
    x = 2.0*pi*r[None,:]/lam[:,None]



    # Индивидуальный nmax для каждой (i,j)
    nmax_ij = x + 4.0 * tf.pow(x, 1.0/3.0) + 5.0  # (nlam, nr)
    nmax_ij = tf.cast(tf.math.ceil(nmax_ij), tf.int32)

    m = refrp/refrm
    mx = tf.cast(x,tf.complex128)*m[:,None]
    xc = tf.cast(x,tf.complex128)
    amx = tf.cast(tf.math.abs(mx), tf.int32)



    nd = tf.cast(tf.cast(tf.reduce_max(tf.maximum(nmax_ij,amx)),tf.float64),tf.int32)+15
    nmax_ij = tf.maximum(nmax_ij,amx)+15

    index = tf.cast(tf.range(0,nd+1),tf.complex128)
    indexf = tf.cast(tf.range(0,nd+1),tf.float64)
    index_int = tf.cast(tf.range(0,nd+1),tf.int32)

    D = 1/mx[:,:,None]
    D = tf.repeat(D,nd+1,axis=2)


    D = tf.Variable(D*(index+1))#numpy.array([complex((i+1)/mx) for i in range(nd+1)])

    zeros = tf.zeros([nlam,nr],tf.complex128)

    D[:,:, nd].assign(zeros)
    for i in range(nd,-1,-1):
        D[:,:,i - 1].assign(D[:,:,i - 1] - 1.0 / (D[:,:,i] + D[:,:,i - 1]))


    zeros1 = tf.zeros([nlam,nr, nd+1],tf.float64)
    zerosc = tf.zeros([nlam,nr, nd+1],tf.complex128)
    psi = tf.Variable(zeros1)
    ksi = tf.Variable(zeros1)

    psi[:,:,0].assign(tf.math.sin(x))

    psi[:,:,1].assign(tf.math.sin(x)/x-tf.math.cos(x))
    ksi[:,:,0].assign(tf.math.cos(x))
    ksi[:,:,1].assign(tf.math.cos(x)/x+tf.math.sin(x))

    # Создаём маску валидности для всех n
    n_indices = tf.range(nd + 1, dtype=tf.int32)  # (nd+1,)
    # valid_mask[i,j,n] = True, если n <= nmax_ij[i,j]
    valid_mask = n_indices[None, None, :] <= nmax_ij[:, :, None]  # (nlam, nr, nd+1)

    for i in range(2,nd+1):
        # Вычисляем новые значения
        new_psi = ((2.0 * (i - 1) + 1.0) * psi[:, :, i - 1] / x) - psi[:, :, i - 2]
        new_ksi = ((2.0 * (i - 1) + 1.0) * ksi[:, :, i - 1] / x) - ksi[:, :, i - 2]

        # Маска для текущего n = i: (nlam, nr)
        mask_i = valid_mask[:, :, i]

        # Обновляем ТОЛЬКО те элементы, где n=i допустимо
        # Где маска False — оставляем старое значение (обычно 0)
        psi_update = tf.where(mask_i, new_psi, psi[:, :, i])
        ksi_update = tf.where(mask_i, new_ksi, ksi[:, :, i])

        # Присваиваем
        psi[:, :, i].assign(psi_update)
        ksi[:, :, i].assign(ksi_update)






    eps = tf.complex(psi,-ksi)




    a = tf.Variable(zerosc)
    b = tf.Variable(zerosc)
    psic = tf.cast(psi, tf.complex128)


    ix = index[:,None,None]/xc
    ix = tf.transpose(ix,perm = (1,2,0))

    tmp1 = D/m[:,None,None]+ix
    tmp2 = D*m[:,None,None]+ix
    aa = tf.where(valid_mask[:,:,1:nd+1], (tmp1[:,:,1:] * psic[:,:,1:] - psic[:,:,0:-1]) / (tmp1[:,:,1:] * eps[:,:,1:] - eps[:,:,0:- 1]), 0.0)
    a[:,:,1:nd+1].assign(aa)
    bb = tf.where(valid_mask[:,:,1:nd+1],(tmp2[:,:,1:] * psic[:,:,1:] - psic[:,:,0:-1]) / (tmp2[:,:,1:] * eps[:,:,1:] - eps[:,:,0: - 1]), 0.0)
    b[:,:,1:nd+1].assign(bb)


    #real_part = tf.math.real(b)
    #imag_part = tf.math.imag(b)

    # Поиск NaN в реальной части
    #real_nan_mask = tf.math.is_nan(real_part)
    #real_nan_count = tf.reduce_sum(tf.cast(real_nan_mask, tf.int32))
    #imag_nan_mask = tf.math.is_nan(imag_part)
    #imag_nan_count = tf.reduce_sum(tf.cast(imag_nan_mask, tf.int32))
    #print("=========-----------============")
    #print(imag_nan_count)
    #print(real_nan_count)



    qexta = tf.math.real((a+b)*(2*index+1))
    qext = tf.reduce_sum(qexta,axis=2)
    qscta = (tf.math.abs(a)**2+tf.math.abs(b)**2)*(2.0*indexf+1.0)
    qsct = tf.reduce_sum(qscta,axis=2)
    qsctpica  = (a-b)*tf.cast(1 - 2*tf.math.mod(index_int , 2), tf.complex128)*(2*index+1)

    qsctpi = abs(tf.reduce_sum(qsctpica,axis=2))**2/(x**2)

    qext=qext*2.0/x**2.0
    qsct=qsct*2.0/x**2.0
    sct=qsct*(r**2)*pi*1e-8
    ext=qext*(r**2)*pi*1e-8


    spi1 = qsctpi*(r**2)*pi*1e-8
    return ext,sct, spi1




def compare_mie_implementations_grid(wavelengths_nm, radii_um, nang=180):
    """
    Сравнение трёх реализаций теории Ми:
    - miesctopt (NumPy)
    - PyMieScatt (эталон)
    - mieosct_tensor (TensorFlow, векторизованный)

    Предполагается, что mieosct_tensor принимает:
        rl.lam: вектор длин волн (в микрометрах)
        r: вектор радиусов (в микрометрах)
    и возвращает тензоры формы (len(wavelengths), len(radii)).
    """
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=np.float64)
    radii_um = np.asarray(radii_um, dtype=np.float64)
    Nλ = len(wavelengths_nm)
    Nr = len(radii_um)
    total_points = Nλ * Nr

    print(f"Сравнение: {Nλ} длин волн × {Nr} радиусов = {total_points} точек\n")

    # ================================
    # 1. TensorFlow расчёт
    # ================================
    print("→ TensorFlow (mieosct_tensor)...")
    rl = lam_refr(Nλ)
    rl.lam[:] = wavelengths_nm / 1000.0  # нм → мкм
    rl.refrp[:] = 1.59 + 0.0j


    with tf.device('/CPU:0'):
        ext_tf, sct_tf, spi1_tf = mieosct_tensor(rl=rl, r=radii_um)

    ext_tf = ext_tf.numpy()
    sct_tf = sct_tf.numpy()
    spi1_tf = spi1_tf.numpy()

    # Обработка возможных 1D случаев
    if ext_tf.ndim == 1:
        if Nλ == 1:
            ext_tf = ext_tf[None, :]
            sct_tf = sct_tf[None, :]
            spi1_tf = spi1_tf[None, :]
        elif Nr == 1:
            ext_tf = ext_tf[:, None]
            sct_tf = sct_tf[:, None]
            spi1_tf = spi1_tf[:, None]

    assert ext_tf.shape == (Nλ, Nr), f"TF shape {ext_tf.shape} != ({Nλ}, {Nr})"

    factor = np.pi * radii_um[None, :]**2  # (1, Nr)
    Q_sca_tf = (sct_tf / factor) * 1e8
    Q_ext_tf = (ext_tf / factor) * 1e8
    Q_back_tf = (spi1_tf / factor) * 1e8


    # ================================
    # 2. NumPy и PyMieScatt
    # ================================
    print("→ NumPy (miesctopt) и PyMieScatt...")

    Q_sca_np = np.full((Nλ, Nr), np.nan)
    Q_ext_np = np.full((Nλ, Nr), np.nan)
    Q_back_np = np.full((Nλ, Nr), np.nan)
    Q_sca_pymie = np.full((Nλ, Nr), np.nan)
    Q_ext_pymie = np.full((Nλ, Nr), np.nan)
    Q_back_pymie = np.full((Nλ, Nr), np.nan)

    angles_rad = np.deg2rad(np.linspace(0, 180, nang))

    for i, wl in enumerate(wavelengths_nm):
        for j, r in enumerate(radii_um):
            # NumPy
            res = miesctopt(
                refrp=1.59,
                refrm=1.0,
                nang=nang,
                ang=angles_rad,
                r=r,
                lam=wl / 1000.0
            )
            f = np.pi * r**2
            Q_sca_np[i, j] = res.sct / f * 1e8
            Q_ext_np[i, j] = res.ext / f * 1e8
            Q_back_np[i, j] = res.spi1 / f * 1e8

            # PyMieScatt
            diam = 2 * r * 1000  # мкм → нм
            out = ps.MieQ(m=1.59+0j, wavelength=wl, diameter=diam, nMedium=1.0)
            if len(out) >= 6:
                qext, qsca, _, _, _, qback = out[:6]
            else:
                qext = qsca = qback = 0.0
            Q_sca_pymie[i, j] = qsca
            Q_ext_pymie[i, j] = qext
            Q_back_pymie[i, j] = qback

    # ================================
    # 3. Текстовый вывод — по всем точкам
    # ================================
    print("\n" + "="*90)
    print(f"{'Точка':<6} {'λ (нм)':<8} {'r (мкм)':<8} {'Q_sca':<24} {'Q_ext':<24} {'Q_back':<12}")
    print("="*90)

    for idx in range(total_points):
        i = idx // Nr
        j = idx % Nr
        wl = wavelengths_nm[i]
        r = radii_um[j]

        qs_pymie = Q_sca_pymie[i, j]
        qe_pymie = Q_ext_pymie[i, j]
        qb_pymie = Q_back_pymie[i, j]

        qs_np = Q_sca_np[i, j]
        qe_np = Q_ext_np[i, j]
        qb_np = Q_back_np[i, j]

        qs_tf = Q_sca_tf[i, j]
        qe_tf = Q_ext_tf[i, j]
        qb_tf = Q_back_tf[i, j]

        d_qs_np = abs(qs_np - qs_pymie)
        d_qb_np = abs(qb_np - qb_pymie)
        d_qs_tf = abs(qs_tf - qs_pymie)
        d_qb_tf = abs(qb_tf - qb_pymie)

        print(f"{idx+1:<6} {wl:<8.0f} {r:<8.2f} "
              f"NP:{qs_np:7.4f}±{d_qs_np:.1e}  TF:{qs_tf:7.4f}±{d_qs_tf:.1e}  Py:{qs_pymie:7.4f} | "
              f"Qb: NP:{qb_np:.2e}±{d_qb_np:.1e}  TF:{qb_tf:.2e}±{d_qb_tf:.1e}  Py:{qb_pymie:.2e}")

    # Сводка по максимальным ошибкам
    dQ_sca_np_max = np.abs(Q_sca_np - Q_sca_pymie).max()
    dQ_back_np_max = np.abs(Q_back_np - Q_back_pymie).max()
    dQ_sca_tf_max = np.abs(Q_sca_tf - Q_sca_pymie).max()
    dQ_back_tf_max = np.abs(Q_back_tf - Q_back_pymie).max()

    print("\n" + "="*60)
    print("МАКСИМАЛЬНЫЕ АБСОЛЮТНЫЕ РАСХОЖДЕНИЯ:")
    print(f"NumPy:      ΔQ_sca = {dQ_sca_np_max:.2e},   ΔQ_back = {dQ_back_np_max:.2e}")
    print(f"TensorFlow: ΔQ_sca = {dQ_sca_tf_max:.2e},   ΔQ_back = {dQ_back_tf_max:.2e}")
    print("="*60)

    # ================================
    # 4. Графики
    # ================================
    # Преобразуем в "длинный" формат
    WL, R = np.meshgrid(wavelengths_nm, radii_um, indexing='ij')
    df = pd.DataFrame({
        'wl': WL.ravel(),
        'r': R.ravel(),
        'Q_sca_pymie': Q_sca_pymie.ravel(),
        'Q_sca_np': Q_sca_np.ravel(),
        'Q_sca_tf': Q_sca_tf.ravel(),
        'Q_ext_pymie': Q_ext_pymie.ravel(),
        'Q_ext_np': Q_ext_np.ravel(),
        'Q_ext_tf': Q_ext_tf.ravel(),
        'Q_back_pymie': Q_back_pymie.ravel(),
        'Q_back_np': Q_back_np.ravel(),
        'Q_back_tf': Q_back_tf.ravel(),
    })

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Q_sca ---
    axes[0].scatter(df['Q_sca_pymie'], df['Q_sca_np'], alpha=0.7, label='NumPy', s=60)
    axes[0].scatter(df['Q_sca_pymie'][:-1], df['Q_sca_tf'][:-1], alpha=0.7, label='TensorFlow (кроме последней)', s=60)
    if total_points > 1:
        axes[0].scatter(df['Q_sca_pymie'].iloc[-1], df['Q_sca_tf'].iloc[-1],
                        color='red', s=100, label='TensorFlow (последняя точка)', zorder=5)
    lims = [min(df['Q_sca_pymie'].min(), df['Q_sca_np'].min(), df['Q_sca_tf'].min()) - 0.1,
            max(df['Q_sca_pymie'].max(), df['Q_sca_np'].max(), df['Q_sca_tf'].max()) + 0.1]
    axes[0].plot(lims, lims, 'k--', lw=1, label='Идеальное совпадение')
    axes[0].set_xlim(lims)
    axes[0].set_ylim(lims)
    axes[0].set_xlabel('Q_sca (PyMieScatt)')
    axes[0].set_ylabel('Q_sca (реализации)')
    axes[0].set_title('Коэффициент рассеяния $Q_{sca}$')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Q_ext ---
    axes[1].scatter(df['Q_ext_pymie'], df['Q_ext_np'], alpha=0.7, label='NumPy', s=60)
    axes[1].scatter(df['Q_ext_pymie'][:-1], df['Q_ext_tf'][:-1], alpha=0.7, label='TensorFlow (кроме последней)', s=60)
    if total_points > 1:
        axes[1].scatter(df['Q_ext_pymie'].iloc[-1], df['Q_ext_tf'].iloc[-1],
                        color='red', s=100, label='TensorFlow (последняя точка)', zorder=5)
    lims = [min(df['Q_ext_pymie'].min(), df['Q_ext_np'].min(), df['Q_ext_tf'].min()) - 0.1,
            max(df['Q_ext_pymie'].max(), df['Q_ext_np'].max(), df['Q_ext_tf'].max()) + 0.1]
    axes[1].plot(lims, lims, 'k--', lw=1)
    axes[1].set_xlim(lims)
    axes[1].set_ylim(lims)
    axes[1].set_xlabel('Q_ext (PyMieScatt)')
    axes[1].set_ylabel('Q_ext (реализации)')
    axes[1].set_title('Коэффициент ослабления $Q_{ext}$')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # --- Q_back ---
    axes[2].scatter(df['Q_back_pymie'], df['Q_back_np'], alpha=0.7, label='NumPy', s=60)
    axes[2].scatter(df['Q_back_pymie'][:-1], df['Q_back_tf'][:-1], alpha=0.7, label='TensorFlow (кроме последней)', s=60)
    if total_points > 1:
        axes[2].scatter(df['Q_back_pymie'].iloc[-1], df['Q_back_tf'].iloc[-1],
                        color='red', s=100, label='TensorFlow (последняя точка)', zorder=5)
    lims = [0, max(df['Q_back_pymie'].max(), df['Q_back_np'].max(), df['Q_back_tf'].max()) * 1.1]
    axes[2].plot(lims, lims, 'k--', lw=1)
    axes[2].set_xlim(lims)
    axes[2].set_ylim(lims)
    axes[2].set_xlabel('Q_back (PyMieScatt)')
    axes[2].set_ylabel('Q_back (реализации)')
    axes[2].set_title('Коэффициент обратного рассеяния $Q_{back}$')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return df


# ======================================================
# Запуск
# ======================================================

# Задайте параметры
wavelengths = [400, 550]  # нм
radii = [0.1, 1.0, 12.0 ]   # мкм

# Запустите сравнение
#df_results = compare_multiple_parameters(wavelengths, radii)

# ======================================================
# Запуск c tensorflow
# ======================================================

df_results = compare_mie_implementations_grid(wavelengths, radii)
