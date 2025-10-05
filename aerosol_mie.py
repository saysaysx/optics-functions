#MIT License

#Copyright (c) 2025 saysaysx


import tensorflow as tf
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
        self.ind = numpy.zeros(n)
        self.sctang = numpy.zeros(n)
        self.spi1 = 0.0 

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

class lam_refr:
    def __init__(self,nlam):
        self.nlam = nlam
        self.lam = numpy.ones(nlam)
        self.refrp = numpy.ones(nlam,dtype=complex)
        self.refrm = numpy.ones(nlam,dtype=complex)
        self.refrm[:] = 1.0+0.0j

def miesct_tensor(rl=lam_refr(1),ang=numpy.array([0,pi/2]), r = numpy.array([1])):
    nlam = rl.nlam
    nr = r.shape[0]
    nang = ang.shape[0]

    lam = tf.cast(rl.lam,tf.float64)
    r = tf.cast(r,tf.float64)
    refrp = tf.cast(rl.refrp,tf.complex128)
    refrm = tf.cast(rl.refrm,tf.complex128)
    # size nlam*nr
    x = 2.0*pi*r[None,:]/lam[:,None]

    nmax = tf.reduce_max(x+4.0*x**(1.0/3.0)+2.0)
    nmax = tf.cast(nmax,tf.int32)


    m = refrp/refrm
    mx = tf.cast(x,tf.complex128)*m[:,None]
    xc = tf.cast(x,tf.complex128)

    amx = tf.math.abs(mx)
    maxmx = tf.cast(tf.reduce_max(amx),tf.int32)


    nd = int(tf.maximum(nmax,maxmx)/2)+2
    print(nmax)
    print(maxmx)


    index = tf.cast(tf.range(0,nd+1),tf.complex128)
    indexf = tf.cast(tf.range(0,nd+1),tf.float64)

    D = 1/mx[:,:,None]
    D = tf.repeat(D,nd+1,axis=2)


    D = tf.Variable(D*(index+1))#numpy.array([complex((i+1)/mx) for i in range(nd+1)])
    print(D.shape)
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

    for i in range(2,nd+1):
        psi[:,:,i].assign(((2.0*(i-1)+1.0)*psi[:,:,i-1]/x)-psi[:,:,i-2])
        ksi[:,:,i].assign(((2.0*(i-1)+1.0)*ksi[:,:,i-1]/x)-ksi[:,:,i-2])
    print(ksi[0,0,:])



    eps = tf.complex(psi,-ksi)

    a = tf.Variable(zerosc)
    b = tf.Variable(zerosc)
    psic = tf.cast(psi, tf.complex128)


    ix = index[:,None,None]/xc
    ix = tf.transpose(ix,perm = (1,2,0))

    tmp1 = D/m[:,None,None]+ix
    tmp2 = D*m[:,None,None]+ix



    a[:,:,1:nd+1].assign((tmp1[:,:,1:] * psic[:,:,1:] - psic[:,:,0:-1]) / (tmp1[:,:,1:] * eps[:,:,1:] - eps[:,:,0:- 1]))
    b[:,:,1:nd+1].assign((tmp2[:,:,1:] * psic[:,:,1:] - psic[:,:,0:-1]) / (tmp2[:,:,1:] * eps[:,:,1:] - eps[:,:,0: - 1]))

    zcnang = tf.zeros([nlam,nr, nang],tf.complex128)
    #zfnang = tf.zeros([nlam,nr, nang],tf.float64)

    #zfnangnd = tf.zeros([nlam,nr, nang, nd+1],tf.float64)
    zcnangnd = tf.zeros([nlam,nr, nang, nd+1],tf.complex128)

    pq = tf.Variable(zcnangnd)
    tq = tf.Variable(zcnangnd)

    s1a = tf.Variable(zcnangnd)
    s2a = tf.Variable(zcnangnd)

    pq[:,:,:,0].assign(zcnang)
    pq[:,:,:,1].assign(zcnang+1.0)
    tq[:,:,:,1].assign(tf.cast(tf.math.cos(ang),tf.complex128)*(zcnang+1.0))
    for i in range(2,nd+1):
        pq[:,:,:,i].assign(((2.0*i-1.0)/(i-1.0))*tq[:,:,:,1]*pq[:,:,:,i-1]-i*pq[:,:,:,i-2]/(i-1.0))
        tq[:,:,:,i].assign(i*tq[:,:,:,1]*pq[:,:,:,i]-(i+1.0)*pq[:,:,:,i-1])

    for i in range(1,nd+1):
        iv = (2.0*i+1.0)/((i+1)*i)
        s1a[:,:,:,i].assign(iv*(a[:,:,i, None]*pq[:,:,:,i]+b[:,:,i,None]*tq[:,:,:,i]))
        s2a[:,:,:,i].assign(iv*(a[:,:,i, None]*tq[:,:,:,i]+b[:,:,i, None]*pq[:,:,:,i]))

    s1 = tf.reduce_sum(s1a,axis=3)
    s2 = tf.reduce_sum(s2a,axis=3)



    qexta = tf.math.real((a+b)*(2*index+1))
    qext = tf.reduce_sum(qexta,axis=2)
    qscta = (tf.math.abs(a)**2+tf.math.abs(b)**2)*(2.0*indexf+1.0)
    qsct = tf.reduce_sum(qscta,axis=2)
    qsctpica  = (a-b)*((-1)**index)*(2*index+1)

    qsctpi = abs(tf.reduce_sum(qsctpica,axis=2))**2/(x**2)

    qext=qext*2.0/x**2.0
    qsct=qsct*2.0/x**2.0
    sct=qsct*(r**2)*pi*1e-8
    ext=qext*(r**2)*pi*1e-8

    ind = tf.math.abs(s1)**2+tf.math.abs(s2)**2

    sinang = tf.math.sin(ang)
    inds = (ind[:,:,1:]*sinang[1:]+ind[:,:,0:-1]*sinang[0:-1])*0.5*(ang[1:]-ang[0:-1])
    indsv = tf.reduce_sum(inds,axis=2)
    indnew = inds/indsv[:,:,None]
    spi = indnew[:,:,-1]*sct*2
    sctang = indnew*sct[:,:,None]*2
    spi1 = qsctpi*(r**2)*1e-12
    return ext,sct,spi,sctang, spi1






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


lamm = numpy.array([0.355, 0.470, 0.528, 0.532, 0.630, 0.850, 0.940 ])
radsm = numpy.array([10.1*i/55+0.10 for i in range(55)])
rlobj = lam_refr(len(lamm))
rlobj.refrp[:] = 1.3+0.0001j
rlobj.lam = lamm

ext,sct,spi,sctang,spi1 = miesct_tensor(rl = rlobj,  ang = numpy.linspace(0,pi,360), r = radsm)

import matplotlib.pyplot as plt

plt.plot(radsm, spi[0])
plt.plot(radsm, spi1[0])

plt.show()
