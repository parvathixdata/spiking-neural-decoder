#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import gridspec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[19]:


df = np.loadtxt('DataSetCoursework_RAJ_PAR.txt')
df=pd.DataFrame(df)
spikes = np.loadtxt('DataSetCoursework_RAJ_PAR.txt')


# In[20]:


df


# In[4]:


def LeakyIF_3(input_currents=None, duration=0.2, dt=0.0001):
    # PARAMETER SETUP
    # 1. Fixed parameters: Membrane constants (all IS units)
    tau = 0.020  # Membrane time constant in seconds (R x C in the model)
    R = 3e7  # Membrane resistance in ohms
    U_rest = -0.07  # Resting potential of the cell in volts (also reset potential)
    theta = -0.030  # Threshold of depolarization for a spike to trigger in volts
    spikeVolt = 0.1  # Spike value in volts
    backgroundI = 3e-9  # Noise input background in Amp, present all the time from the background input in the medium
    tau_adapt = 0.3  # How fast the adaptive threshold decays to the initial "theta" value when no spikes
    increase_threshold = 0.012  # In volts, this is how much the threshold increases after a spike

    # 1.3 Simulation parameters
    if input_currents is None:
        input_currents = {}  # Initialize as an empty dictionary
        U_0 = 0.3  # In volts
        I_0 = U_0 / R
        psc = np.arange(0, duration, dt)
    else:
        U_0 = input_currents.get('U_0', 0.3)
        I_0 = U_0 / R
        psc = input_currents.get('psc', np.arange(0, duration, dt))

    ipsc = input_currents.get('ipsc', [])

    # MODEL

    # 1-Initializations
    n_pcs = len(psc)
    n_ipcs = len(ipsc)
    index_pscs = np.round(np.array(psc) / dt)
    index_ipscs = np.round(np.array(ipsc) / dt)
    n_steps = round(duration / dt)
    U = np.zeros(n_steps + 1)
    U_plot = np.zeros(n_steps + 1)
    U[0] = U_rest
    U_plot[0] = U_rest
    I = np.zeros(n_steps + 1)
    t_spike = 0
    n_spikes = 0
    time = np.linspace(0, duration, n_steps + 1)
    randI = backgroundI * np.random.normal(0, 1, n_steps)

    # Move arp outside the if block and set its value inside the block
    arp = 0.008 if 'arp' in locals() else 0.0  # Set a default value for arp

    theta_adapt = np.full(n_steps + 1, theta)

    # Use the real spike data to estimate parameters
    if spikes.size > 0:
        real_spike_times = spikes[spikes < duration]
        avg_spike_rate = len(real_spike_times) / duration
        estimated_tau = 1 / (2 * np.pi * avg_spike_rate)  # Estimated from average spike rate
        estimated_increase_threshold = np.std(np.diff(real_spike_times))

        tau = estimated_tau if estimated_tau > 0 else tau
        increase_threshold = estimated_increase_threshold if estimated_increase_threshold > 0 else increase_threshold

    # Simulate the spiking model
    for i in range(1, n_steps):  # Update range to n_steps - 1
        for k in range(n_pcs):
            if i == index_pscs[k]:
                I[i] += I_0

        for x in range(n_ipcs):
            if i == index_ipscs[x]:
                I[i] -= I_0

        # Leaky integrate-and-fire dynamics
        dU = (dt / tau) * (U_rest - U[i] + I[i] * R + randI[i] * R)
        U[i + 1] = U[i] + dU
        U_plot[i + 1] = U[i + 1]

        # Spike threshold adaptation
        theta_adapt[i] = theta_adapt[i - 1] - (theta_adapt[i - 1] - theta) / tau_adapt * dt

        # Spike condition
        if U[i] >= theta_adapt[i]:
            U[i + 1] = U_rest  # Reset membrane potential
            theta_adapt[i + 1] += increase_threshold  # Increase threshold after spike
            n_spikes += 1
            t_spike = i * dt
    # Adjust the length of U_plot to match the length of time
    U_plot = U_plot[:n_steps + 1]  # Use n_steps instead of len(time)
    # Ensure time array is defined correctly up to n_steps
    time = np.linspace(0, duration, n_steps+1)

    # PLOTTING
    plt.figure()
    plt.plot(time, U_plot, color='black')
    plt.hlines(theta, 0, time[-1], color='red', linestyle='--')
    plt.title('Leaky I&F with background current, refractory period, excit/inhib synapses '
              'and adaptive threshold. Rest value: {:.2f} mV. Refractory period {:.2f} ms'
              .format(theta * 1000, arp * 1000))
    plt.xlabel('time (s)')
    plt.ylabel('voltage (U), number of spikes={}'.format(n_spikes))
    plt.grid(True)

    plt.figure()
    plt.plot(time, theta_adapt, color='red')
    plt.hlines(theta, 0, time[-1], color='red', linestyle='--')
    plt.title('Rest value of threshold: {:.2f} mV. Refractory period {:.2f} ms'
              .format(theta * 1000, arp * 1000))
    plt.xlabel('time (s)')
    plt.ylabel('Spike threshold (U), number of spikes={}'.format(n_spikes))
    plt.grid(True)

    plt.show()
    # Compare real spikes with simulated spikes
    plt.figure()
    plt.plot(time, U_plot, color='black')
    plt.scatter(real_spike_times, np.ones_like(real_spike_times) * spikeVolt, color='blue', marker='o', label='Real Spikes')
    plt.scatter(time[U_plot > theta], np.ones_like(U_plot[U_plot > theta]) * spikeVolt, color='yellow', marker='x', label='Simulated Spikes')
    plt.hlines(theta, 0, time[-1], color='red', linestyle='--')
    plt.title('Leaky I&F with background current, refractory period, excit/inhib synapses '
              'and adaptive threshold. Rest value: {:.2f} mV. Refractory period {:.2f} ms'
              .format(theta * 1000, arp * 1000))
    plt.xlabel('time (s)')
    plt.ylabel('voltage (U), number of spikes={}'.format(n_spikes))
    plt.legend()
    plt.grid(True)
    plt.show()


# In[5]:


#Adaptability
dt=0.0001;#In secs(1e-4)
input_currents = {'I_0': 1e-8} #In amps
psc = np.concatenate([np.arange(0.05, 0.056, dt), np.arange(0.06, 0.063, dt), np.arange(0.1, 0.105, dt),np.arange(0.105, 0.109, dt),np.arange(0.10801, 0.131, dt)])
#'psc': np.array([0.05, 0.1, 0.15])
input_currents['psc'] = psc
#LeakyIF_1(input_currents,0.2,dt)
#LeakyIF_2(input_currents,0.2,dt)
LeakyIF_3(input_currents,0.2,dt)


# In[6]:


def estimate_firing_rates(spikes, window_size=5, bin_size=0.01):
    n, m = spikes.shape
    times = np.arange(0, n * bin_size, bin_size)
    window_steps = int(window_size / bin_size)

    rates = np.zeros((n, m))

    for j in range(m):
        smoothed = []
        for i in range(n - window_steps + 1):
            spikes_one_neuron = spikes[i:i + window_steps, j]
            rate_in_this_window = np.sum(spikes_one_neuron) * (1 / window_size)
            smoothed.append(rate_in_this_window)

        if smoothed:
            smoothed.extend([smoothed[-1]] * (window_steps - 1))
        else:
            smoothed.extend([0] * (window_steps - 1))

        rates[:, j] = np.array(smoothed)[:n]

    return rates, times

def plot_firing_rates(ax, times, rates, color='b', title='', ylabel='Spikes'):
    ax.plot(times, rates, color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
def plot_firing_rate_sec(ax, times, rates, color='r', title=''):
    ax.plot(times, rates, color)
    ax.set_title(title)
    ax.set_ylabel('Spikes/sec')

# Load real neural data
df = np.loadtxt("DataSetCoursework_DSO.txt")
trial_num = df[:, -1]
signal = df[:, -2]
real_spikes = df[:, :-2]

# Estimate firing rates for real data
window_size = 5
bin_size = 0.01
real_rates, times = estimate_firing_rates(real_spikes, window_size, bin_size)
def plot_real_data(axs, times, real_rates):
    for j in range(real_rates.shape[1]):
        ax_spikes = axs[0, j]
        ax_rate = axs[1, j]

        plot_firing_rates(ax_spikes, times, real_rates[:, j], title=f'Real Neuron {j + 1}')
        plot_firing_rate_sec(ax_rate, times, real_rates[:, j], title=f'Real Neuron {j + 1} rate')

    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.show()

# Plot real data
fig, axs = plt.subplots(nrows=2, ncols=6, figsize=(30, 5))
plot_real_data(axs, times, real_rates)


# In[21]:


df = np.loadtxt('DataSetCoursework_RAJ_PAR.txt')
df = pd.DataFrame(df)
X = df.iloc[:, 0:6]
y = df.iloc[:, 6]

# Splitting into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Training the classification models with default parameters
svc = SVC()
lda = LinearDiscriminantAnalysis()
rfc = RandomForestClassifier()
ada = AdaBoostClassifier()

svc.fit(X_train, y_train)
lda.fit(X_train, y_train)
rfc.fit(X_train, y_train)
ada.fit(X_train, y_train)

# Predictions on the test set
svc_y_pred = svc.predict(X_test)
lda_y_pred = lda.predict(X_test)
rfc_y_pred = rfc.predict(X_test)
ada_y_pred = ada.predict(X_test)

# Display models results using classification metrics for multiclass
models = ['SVC', 'LDA', 'RandomForest', 'AdaBoost']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']

results = []

for model, y_pred in zip(models, [svc_y_pred, lda_y_pred, rfc_y_pred, ada_y_pred]):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    results.append([accuracy, precision, recall, f1])

df_results = pd.DataFrame(results, index=models, columns=metrics)
print("Classification Metrics:")
print(df_results)

# Plot confusion matrices using seaborn
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

for ax, model, y_pred in zip(axes.flatten(), models, [svc_y_pred, lda_y_pred, rfc_y_pred, ada_y_pred]):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'Confusion Matrix - {model}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

plt.tight_layout()
plt.show()


# In[8]:


print(df.shape)


# In[17]:


unique_values_y = y.unique()
print(unique_values_y)


# In[18]:


df.to_csv('dataset.csv', index=False)

