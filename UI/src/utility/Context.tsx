/*
    Global data context used across components in App.tsx. Contains visualization settings
    (whether to display wire selection, error bars, etc.), X-ray settings (prone vs. supine),
    and other user choices (selected wire and corridor).
*/

import React, { createContext, useState, useMemo, ReactNode } from 'react';

type PatientViewType = 'supine' | 'prone';

export interface DataContextType {
    selectedWire: string;
    selectedCorridor: string;
    patientView: PatientViewType;
    showAnatomy: boolean;
    showWireSelect: boolean,
    showCorridor: boolean,
    showErrBars: boolean;
    showImage: boolean,
    imageEncoding: string;
    setSelectedWire: (wire: string) => void;
    setSelectedCorridor: (corridor: string) => void;
    setPatientView: (view: PatientViewType) => void;
    setImageEncoding: (image: string) => void;
    setShowAnatomy: (show: boolean) => void;
    setShowCorridor: (show: boolean) => void;
    setShowWireSelect: (show: boolean) => void;
    setShowImage: (show: boolean) => void;
    setShowErrBars: (show: boolean) => void;
}

export const Context = createContext<DataContextType>(undefined!);

interface DataProviderProps {
    children: ReactNode;
}

export const DataProvider: React.FC<DataProviderProps> = ({ children }) => {
    const [selectedWire, setSelectedWire] = useState<string>('wire0');
    const [selectedCorridor, setSelectedCorridor] = useState<string>('ramus_right');
    const [patientView, setPatientView] = useState<PatientViewType>('supine');
    const [showAnatomy, setShowAnatomy] = useState<boolean>(true);
    const [showWireSelect, setShowWireSelect] = useState<boolean>(true);
    const [showCorridor, setShowCorridor] = useState<boolean>(true);
    const [showImage, setShowImage] = useState<boolean>(true);
    const [showErrBars, setShowErrBars] = useState<boolean>(false);
    const [imageEncoding, setImageEncoding] = useState<string>('');

    const value = useMemo(() => ({
        selectedWire,
        setSelectedWire,
        selectedCorridor,
        setSelectedCorridor,
        patientView,
        setPatientView,
        showAnatomy,
        setShowAnatomy,
        showWireSelect,
        setShowWireSelect,
        showImage,
        setShowImage,
        showCorridor,
        setShowCorridor,
        showErrBars,
        setShowErrBars,
        imageEncoding,
        setImageEncoding
    }), [selectedWire, selectedCorridor, patientView, showAnatomy, showCorridor, showWireSelect, showImage, showErrBars, imageEncoding]);

    return (
        <Context.Provider value={value}>
            {children}
        </Context.Provider>
    );
};